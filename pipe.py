from clearml import PipelineController

pipe = PipelineController(
  name="try_func", project="eval"
)
pipe.upload_artifact("aa", 4)
pipe.add_parameter(
    name='name',
    description='aaccc', 
    default='abc222'
)
def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return

def step_one():
    print('generate args==============================')
    return 4, 5

def step_two(aa, bb):
    print('step_two print ===========')
    print('aa, bb:  ', aa, bb)

pipe.set_default_execution_queue('gpu')
pipe.add_function_step(
     name='step_one',
     function=step_one,
     function_return=['aa', 'bb'],
)
pipe.add_function_step(
     name='step_two',
     function=step_two,
     function_kwargs=dict(aa='${step_one.aa}', bb='${step_one.bb}'),
)
pipe.add_step(
   name='run_task',
   parents=['step_two', ],
   base_task_id='5bea612ec5254abaa1ab9894a7af2fc8',
   parameter_override={
     'Args/aa': '${step_one.id}',
     'Args/bb': '${pipeline.name}',
     'Args/cc': 'yaxixi',
   },
#    pre_execute_callback=pre_execute_callback_example,
#    post_execute_callback=post_execute_callback_example,
)
pipe.start('gpu')