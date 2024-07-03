import subprocess

# 定义你的sbatch脚本的路径
sbatch_script = "/work/09735/yichao/ls6/zhilian/jerry_mamba/mamba_params.sh"

# 定义sbatch命令的完整路径
sbatch_command = "/usr/bin/sbatch"

# 构建完整的命令
command = [sbatch_command, sbatch_script]

# 执行命令
try:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e)