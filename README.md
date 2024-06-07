# 1 本实例实现一个Paillier半同态加密库, 基于c
1. 首先安装gmp库, GMP库实现了任意精度大数的运算.

方法1: 直接安装

    sudo apt-get install libgmp3-dev

方法2: 从 https://gmplib.org 下载最新版本gmp

    wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.gz

    tar -zxvf gmp-6.3.0.tar.gz

    cd gmp-6.3.0

    ./configure --enable-cxx

    make

    make check
    
    sudo make install

# 2 测试gmp库是否安装成功

    gcc -o gmp-test gmp-test.c -lgmp -lm

# 3 测试Paillier算法运行速度
