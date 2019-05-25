from typing import List, Dict 

import numpy as np

from overrides import overrides
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise" : premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

    @overrides
    def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Gets the gradients of the loss with respect to the input and
        returns them normalized and sanitized.  
        """
        return sanitize(self._normalize(self.get_gradients(self.get_model_predictions(inputs))[0]))

    @overrides
    def predictions_to_labels(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        label = np.argmax(outputs['label_logits'])
        # We can skip indexing since we already have the integer representations
        # of the strings ("entailment", etc.)
        instance.add_field('label', LabelField(int(label), skip_indexing=True))
        return [instance]