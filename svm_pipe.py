from clearml import PipelineController

pipe = PipelineController(
  name="chain_svm", project="paper"
)

pipe.add_parameter(
    name='seed',
    description='random seed',
    default=33
)
pipe.add_parameter(
    name='fraction',
    description='stratification fraction', 
    default=0.9
)
pipe.add_parameter(
    name='ratio',
    description='split ratio', 
    default=1
)
pipe.add_parameter(
    name='d_name',
    description='dataset name', 
    default='chain'
)
pipe.add_parameter(
    name='d_project',
    description='dataset project name', 
    default='paper'
)
pipe.set_default_execution_queue('gpu')
pipe.add_step(
   name='gen_data',
   base_task_id='df6039712bbb469ea711b676cde5791b',
   parameter_override={
     'Args/ratio': '${pipeline.ratio}',
     'Args/fraction': '${pipeline.fraction}',
     'Args/seed': '${pipeline.seed}',
     'Args/d_name': '${pipeline.d_name}',
     'Args/d_project': '${pipeline.d_project}',
   },
)
pipe.add_step(
   name='train_svm',
   parents=['gen_data', ],
   base_task_id='124487d4dabb448da7ecdcfb7507055a',
   parameter_override={
     'Args/train_url': '${gen_data.artifacts.train.url}',
     'Args/val_url': '${gen_data.artifacts.val.url}',
   },
)
pipe.add_step(
   name='eval',
   parents=['gen_data', 'train_svm'],
   base_task_id='6495d3d5377749b4a7115b86f4fa9db1',
   parameter_override={
     'Args/model_url': '${train_svm.artifacts.ckpt.url}',
     'Args/val_url': '${gen_data.artifacts.val.url}',
   },
)
pipe.start()