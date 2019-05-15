from typing import List, Dict 

import numpy as np

from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import IndexField, SequenceField
from allennlp.predictors.predictor import Predictor

@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Gets the gradients of the loss with respect to the input and
        returns them normalized and sanitized.  
        """
        return sanitize(self._normalize(self.get_gradients(self.get_model_predictions(inputs))))

    @overrides
    def predictions_to_labels(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        if 'best_span' in outputs: 
            span_start_label = outputs['best_span'][0]
            span_end_label = outputs['best_span'][1]
            instance.add_field('span_start', IndexField(int(span_start_label), SequenceField()))
            instance.add_field('span_end', IndexField(int(span_end_label), SequenceField()))
        elif 'span' in outputs: 
            span_start_label = outputs['span'][0]
            span_end_label = outputs['span'][1]
            instance.add_field('span_start', IndexField(int(span_start_label), SequenceField()))
            instance.add_field('span_end', IndexField(int(span_end_label), SequenceField()))

        return [instance]
