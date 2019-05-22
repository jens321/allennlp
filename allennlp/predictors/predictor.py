from typing import List, Iterator, Dict 
import json
from contextlib import contextmanager

import numpy as np
import torch 
import math 

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'atis_parser' : 'atis_parser',
        'basic_classifier': 'text_classifier',
        'biaffine_parser': 'biaffine-dependency-parser',
        'bidaf': 'machine-comprehension',
        'bidaf-ensemble': 'machine-comprehension',
        'bimpm': 'textual-entailment',
        'constituency_parser': 'constituency-parser',
        'coref': 'coreference-resolution',
        'crf_tagger': 'sentence-tagger',
        'decomposable_attention': 'textual-entailment',
        'dialog_qa': 'dialog_qa',
        'event2mind': 'event2mind',
        'naqanet': 'machine-comprehension',
        'simple_tagger': 'sentence-tagger',
        'srl': 'semantic-role-labeling',
        'quarel_parser': 'quarel-parser',
        'wikitables_mml_parser': 'wikitables-parser'
}

class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + "\n"

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Uses the gradients from :func:`get_gradients` to provide 
        normalized interpretations for specific models. 

        Raises
        ------
        NotImplementedError
            This method should be overriden by the specific predictor class of each
            predictor.
        """
        raise NotImplementedError

    def attack_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Uses the gradients from :func:`get_gradients` to provide 
        adversarial attacks for specific models. 
        Raises
        ------
        NotImplementedError
            This method should be overriden by the specific predictor class of each
            predictor.
        """
        raise NotImplementedError

    def get_model_predictions(self, inputs: JsonDict) -> List[Instance]:
        """
        TODO
        """
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        # Predictions to labels is specific to each predictor,
        # but get_gradients need to be on base class => needs reworking! 
        new_instances = self.predictions_to_labels(instance, outputs)
        return new_instances

    def get_gradients(self, instances: List[Instance]) -> Dict[str, np.ndarray]:
        """
        Gets the gradients of the loss with respect to the model inputs. 
        Parameters
        ----------
        instances: List[Instance]
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of gradient entries for each input fed into the model.
            The keys have the form ``{grad_input_1: ..., grad_input_2: ... }``
            up to the number of inputs given. 
        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks. 
        """
        self._register_hooks()

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)

        # Using forward_on_instances converts the output into numpy arrays
        # instead of keeping the original torch Tensors, so rn we're using the 
        # forward function instead. 
        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))
        print('OUTPUTS', outputs)

        loss = outputs['loss']

        # Zero out computation graph
        self._model.zero_grad()

        # Get the gradients 
        loss.backward()

        # Remove hooks 
        for hook in self.hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(self.extracted_grads):
            key = 'grad_input_' + str(idx + 1)
            # squeeze to remove batch dimension
            grad_dict[key] = grad.squeeze_(0).detach().cpu().numpy()

        return grad_dict 

    def _register_hooks(self):
        """
        Registers a backward hook on the 
        :class:`~allennlp.modules.text_field_embedder.basic_text_field_embbedder.BasicTextFieldEmbedder`
        class. 
        """
        # For multiple inputs, the hook will be called multiple times
        # so we append the incoming gradients to a list
        self.extracted_grads = []
        self.hooks = []

        def hook_layers(module, grad_in, grad_out):
            # grad_in: the gradient with respect to the input of module
            # grad_out: the gradient with respect to the output of module
            # print('grad_in shape', grad_in[0].shape)
            # print('grad_out shape', grad_out[0].shape)
            # print('grad_in values', grad_in)
            # print('grad_out values', grad_out)
            self.extracted_grads.append(grad_in[0])

        def fhook(module, input, output):
            output.mul_(10)

        # Register the hooks
        for module in self._model.modules():
            # We register on BasicTextFieldEmbedder as this seems to 
            # be more general; just "Embedding" does not work
            # for Elmo 
            if isinstance(module, BasicTextFieldEmbedder):
                backward_hook = module.register_backward_hook(hook_layers)
                # forward_hook = module.register_forward_hook(fhook)
                self.hooks.append(backward_hook)
                # self.hooks.append(forward_hook)

    def _normalize(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize the gradients into the range [0,1] for visualization. 

        Notes
        -----
        0.0 is dark red and 1.0 is dark blue. 0.5 is no highlight. 
        This normalization step is very rarely discussed in the 
        literature and the interpretation is actually quite sensitive to it. 
        In the code below, we normalize scores across the words, doing positive 
        and negatives seperately.
        """

        for key, grad in grads.items():
            grad = np.sum(grad, axis=1)
            total_score_pos = 0
            total_score_neg = 0
            for idx, s in enumerate(grad):
                if s < 0:
                    total_score_neg = total_score_neg + math.fabs(s)
                else:
                    total_score_pos = total_score_pos + s
            for idx, s in enumerate(grad):
                if s < 0:
                    # / by 2 to get max of -0.5
                    grad[idx] = (s / total_score_neg) / 2  
                else:
                    grad[idx] = (s / total_score_pos) / 2
            # center scores
            grad = [0.5 + n for n in grad]  
            grads[key] = grad 
        return grads 


    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        .. code-block:: python

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(idx: int):
            def _add_output(mod, _, outputs):
                results[idx] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        for idx, module in enumerate(self._model.modules()):
            if module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        raise NotImplementedError

    def predictions_to_labels(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        """
        Adds labels to the :class:`~allennlp.data.instance.Instance`s passed in. 

        Raises
        ------
        NotImplementedError
            This method should be overriden by the specific predictor class of each
            predictor.
        """
        raise NotImplementedError 

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Converts a list of JSON objects into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by :func:`predict_json`. In order to use this method for
        batch prediction, :func:`_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(cls, archive_path: str, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as running the predictor on the GPU,
        please use `from_archive`.

        Parameters
        ----------
        archive_path The path to the archive.

        Returns
        -------
        A Predictor instance.
        """
        return Predictor.from_archive(load_archive(archive_path), predictor_name)

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("type")
            if not model_type in DEFAULT_PREDICTORS:
                raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
                                         f"Please specify a predictor explicitly.")
            predictor_name = DEFAULT_PREDICTORS[model_type]

        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        model.eval()

        return Predictor.by_name(predictor_name)(model, dataset_reader)
