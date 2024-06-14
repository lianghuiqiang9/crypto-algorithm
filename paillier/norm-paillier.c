#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<gmp.h>
#include<time.h>
int main(){
	 //设置随机数种子
	 clock_t time=clock();
	 gmp_randstate_t grt;
	 gmp_randinit_default(grt); 
	 gmp_randseed_ui(grt, time); 
	 
	 //p、q初始化
	 mpz_t p,q,p1,q1;
 
	 mpz_init(p); 
	 mpz_init(q);
	 mpz_init(p1); 
	 mpz_init(q1);
	 
	 //p、q的的范围在0~2^128-1
	 mpz_urandomb(p, grt, 128);
	 mpz_urandomb(q, grt, 128);
 
	  //生成p,q大素数
	 mpz_nextprime(p, p); 
	 mpz_nextprime(q, q);
	 
	 //判断生成的是否是素数得到的结果为2则确定为大素数
	 int su1,su2;//如果为1则在判断次数内大概率确定是大素数，为0则不是
	 su1=mpz_probab_prime_p(p,100);
	 su2=mpz_probab_prime_p(q,100);
	 printf("判断生成的是否是素数 :p=%d   q=%d\n",su1,su2);
	 //求p，q的乘积 n,以及n的平方n2
	 mpz_t n,n2;
 
	 mpz_init(n);
	 mpz_init(n2);
	 mpz_mul(n,p,q); 
	 mpz_mul(n2,n,n); 
	 
	 //设置g,取值g=n+1
	 mpz_t g,j;
 
	 mpz_init(g);
	 mpz_init_set_ui(j,1);
	 //mpz_urandomb(g, grt, 128);
	 mpz_add(g,n,j);
 
	 //设置明文m
	 mpz_t m;
	 mpz_init_set_str(m,"123456",10);
	 mpz_t r;//设置r,r为随机数
	 mpz_urandomb(r, grt, 128);
 
	 //设置密文c
	 mpz_t c;
 
	 mpz_init(c);
	 //设置密文c
 
	 mpz_powm(c,g,m,n2);
	 mpz_powm(r,r,n,n2);
	 mpz_mul(c,c,r);
	 mpz_mod(c,c,n2);
 
	 //解密过程
	 //先求λ，是p、q的最小公倍数,y3代表λ
	 mpz_t y1,y2,y3;
 
	 mpz_init(y1);
	 mpz_init(y2);
	 mpz_init(y3);
	 
	 mpz_sub(p1,p,j);
	 mpz_sub(q1,q,j);
	 mpz_lcm(y3,p1,q1);//y3代表λ
	 
	 //输出明文m,g
	 //十进制输出是%Zd,十六进制输出是%ZX,folat使用&Ff
	 gmp_printf("明文m = %Zd\n\n", m);
	 //gmp_printf("p = %Zd\n\n", p);
	 //gmp_printf("q = %Zd\n\n", q);
	 //gmp_printf("r = %Zd\n\n", r);
	 //gmp_printf("g = %Zd\n\n", g);
	 //gmp_printf("λ = %Zd\n\n", y3);
	 //输出密文
	 gmp_printf("密文c = %Zd\n\n",c);
	 
	 clock_t start, finish;
	 start = clock();
	 //y1代表c的λ次方摸n平方
	 mpz_powm(y1,c,y3,n2);
	 mpz_sub(y1,y1,j);
	 mpz_div(y1,y1,n);
	 
	 //y2代表g的λ次方摸n平方
	 mpz_powm(y2,g,y3,n2);
	 mpz_sub(y2,y2,j);
	 mpz_div(y2,y2,n);
	 
	 mpz_t x_y;
	 mpz_init(x_y);
	 mpz_invert(x_y,y2,n);//至关重要的一步，取逆
	 
	 mpz_mul(x_y,x_y,y1);
	 mpz_mod(x_y,x_y,n);
	 finish = clock();
	 printf("time is :%f ms\n",(double)(finish - start) * 1000.0 / CLOCKS_PER_SEC);

	 //输出明文
	 gmp_printf("解密得到明文m = %Zd\n\n",x_y);
	 mpz_clear(p);
	 mpz_clear(q);
	 mpz_clear(n);
	 mpz_clear(n2);
	 mpz_clear(p1);
	 mpz_clear(q1);
	 mpz_clear(c);
	 mpz_clear(g);
	 mpz_clear(j);
	 mpz_clear(r);
	 mpz_clear(m);
	 mpz_clear(y2);
	 mpz_clear(y1);
	 mpz_clear(y3);
	 mpz_clear(x_y);
	 
	 return 0;
}