
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

    # structure:
    # {instance_1: grad_dict, instance_2: grad_dict}

    instances_with_grads = dict()
    for idx, instance in enumerate(labeled_instances):
      grads = self.predictor.get_gradients([instance])[0]

      for key, grad in grads.items():
        l2_grad = numpy.linalg.norm(grad, axis=1)
        normalized_grad = normalize_by_total_score(l2_grad)
        grads[key] = normalized_grad 

      instances_with_grads['instance_' + str(idx + 1)] = grads

    print("INSTANCES WITH GRADS")
    print("--------------------")
    print(instances_with_grads)
    return sanitize(instances_with_grads)

    # print('GRADS')
    # print('-----')
    # grads = self.predictor.get_gradients(labeled_instances)[0]
    # print(grads)

    # print('L2-NORM')
    # print('-------')
    # for key, grad in grads.items():
    #   l2_grad = numpy.linalg.norm(grad, axis=1)
    #   normalized_grad = normalize_by_total_score(l2_grad)
    #   grads[key] = normalized_grad
    # print(grads)

    # return sanitize(grads)
