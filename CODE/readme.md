## 代码逻辑结构
|—— test               //测试集（由组委会专家置入）
|—— result.pth              //存放模型文件
|—— data_process.py    //数据预处理文件
|—— model.py       //算法模型文件
|—— train.py           //训练代码文件
|—— test.py            //测试代码文件
|—— requirements.txt   //项目所需依赖包的详细版本号
|—— readme.md          //代码说明文档
|—— 作品说明书.doc
|—— 安装配置说明.doc


## 运行依赖环境
1. 下载并安装Anaconda3-2020.02的Windows 64bit版本
2. 点击电脑左下角Windows图标，找到Anaconda Prompt并运行（若有报错，则尝试以管理员身份运行Anaconda Prompt）
3. 创建名为TEST_CODE的conda环境（环境名称可自由设置，下同），python版本选择3.7，键入命令
`conda create -n TEST_CODE python=3.7`
4. 进入刚创建的TEST_CODE环境，键入命令
`conda activate TEST_CODE`
5. 切换到requirements.txt所在的路径下，一键安装此项目所需的所有依赖包,键入命令
`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`
若有红字 ERROR 报错，则按系统提示重新键入命令。其中，-i后的链接为清华镜像源网址，用此镜像源可加速安装过程。
==****注：详细安装配置过程请查看文档‘安装配置说明.doc’****==


## 启动步骤
1. 启动pycharm，打开项目程序
2. 在pycharm界面的右下角选择解释器，添加刚刚创建的TEST_CODE的环境，稍等片刻让环境加载
3. 直接运行test.py 或
在下方终端输入命令 ：`python test.py`，测试结果 `output.txt` 将保存到test文件夹下
==****注：详细启动步骤请查看文档‘安装配置说明.doc’****==