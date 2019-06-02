
import numpy

from allennlp.common.util import JsonDict, sanitize, normalize_by_total_score
from allennlp.interpretation import Interpreter

@Interpreter.register('simple-gradients-interpreter')
class SimpleGradient(Interpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
    """
    Gets the gradients of the loss with respect to the input and
    returns them normalized and sanitized.  
    """

    # Get labeled instances
    labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

    print('GRADS')
    print('-----')
    grads = self.predictor.get_gradients(labeled_instances)[0]
    print(grads)

    print('L2-NORM')
    print('-------')
    for key, grad in grads.items():
      l2_grad = numpy.linalg.norm(grad, axis=1)
      normalized_grad = normalize_by_total_score(l2_grad)
      grads[key] = normalized_grad
    print(grads)

    return sanitize(grads)
