一. 文件功能：

***1.model.py: 算法与模型框架主要实现区域

2.main.py: 模型训练流程制定与运行区域，添加优化功能提高鲁棒

3.config.py: 模型参数配置区域

4.utils.py: 功能与main.py中存在重复，主要保存优化功能

***5.usb-fm.py: 使用usb提供的算法构建模型，并在此直接训练

***6. torchSSL-fm.py: 使用提供torchSSL的算法构建模型，并在此直接训练

7. visulize.py: 根据日志输出对模型训练过程进行可视化，包括loss与acc 随epoch变化曲线 

二. 文件夹内容：

1.project: 存放模型相关代码

2.output: 存放模型的训练可视化结果以及训练过程各epoch acc/loss情况
	
	/log:存放日志
	· model.log：自己实现的模型

	· usb.log（40/250/400）：usb提供的模型

	· torchSSL.log（40/250/400）: torchSSL提供的模型

	/模型训练过程可视化：存放可视化图像结果