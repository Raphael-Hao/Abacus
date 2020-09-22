这个仓库中我已经把需要的第三方库之类的整合过，额外的库包括：
- 需要自己安装opencv，可以去官方仓库pull，版本为3.4.6
- mxnet有可能会因为静态库过大ar打包错误，可能需要安装另外一个版本的ar，这个原始的安装文件我打包放在外层的tools文件夹下面

编译：
```shell
mkdir build
cd build
make -j$(nproc)
```
编译完成以后可以尝试一下cpp-package/examples下面的`imagenet_inference`示例。需要的symbol，param文件可以在`/home/disk1/cwh/data/models`目录下找到。测试命令如下
```shell
./imagenet_inference --symbol_file /home/disk1/cwh/data/models/resnet/resnet_50-symbol.json --params_file /home/disk1/cwh/data/models/resnet/resnet_50-0000.params --benchmark --gpu
```