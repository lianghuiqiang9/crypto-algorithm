#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<gmp.h>
#include<time.h>
 
//g++ test_1.c -o test1 -lgmp -lm
//./test1
 
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
	 mpz_add(g,n,j);
 
	 //设置明文m
	 mpz_t m;
	 mpz_init_set_str(m,"123456",10);
	 mpz_t r;//设置r,r为随机数
	 mpz_urandomb(r, grt, 128);
 
	 //设置密文c,c1,需要对这两个密文做同态加法
	 mpz_t c;
 
	 mpz_init(c);
	 //设置密文c
 
	 mpz_powm(c,g,m,n2);
	 mpz_powm(r,r,n,n2);
	 mpz_mul(c,c,r);
	 mpz_mod(c,c,n2);	 
	 //输出密文
	 gmp_printf("明文m = %Zd\n\n", m);
	 gmp_printf("密文c = %Zd\n\n",c);	 
	 
	 //中国剩余定理做paillier
	 mpz_sub(p1,p,j);
	 mpz_sub(q1,q,j);
	 mpz_t m_p,m_q,p_2,q_2,mp_y1,mp_y2,mq_y1,mq_y2,p_ni,q_ni,q_p;
	 mpz_init(m_p);
	 mpz_init(q_p);
	 mpz_init(p_ni);
	 mpz_init(q_ni);
	 mpz_init(m_q);
	 mpz_init(p_2);
	 mpz_init(q_2);
	 mpz_init(mp_y1);
	 mpz_init(mp_y2);
	 mpz_init(mq_y1);
	 mpz_init(mq_y2);
	 
	 mpz_mul(p_2,p,p);
	 mpz_mul(q_2,q,q);
	 mpz_invert(p_ni,p,p);
	 mpz_invert(q_ni,q,q);
	 
	 //count time 
	 clock_t start, finish;
	 start = clock();
	 //m_p
	 mpz_powm(mp_y1,c,p1,p_2);
	 mpz_sub(mp_y1,mp_y1,j);
	 mpz_div(mp_y1,mp_y1,p);
	 
	 mpz_powm(mp_y2,g,p1,p_2);
	 mpz_sub(mp_y2,mp_y2,j);
	 mpz_div(mp_y2,mp_y2,p);
	 mpz_invert(mp_y2,mp_y2,p);
	 mpz_mod(mp_y2,mp_y2,p);
	 mpz_mul(mp_y1,mp_y1,mp_y2);
	 mpz_mod(m_p,mp_y1,p);
	 
	 //m_q
	 mpz_powm(mq_y1,c,q1,q_2);
	 mpz_sub(mq_y1,mq_y1,j);
	 mpz_div(mq_y1,mq_y1,q);
	 
	 mpz_powm(mq_y2,g,q1,q_2);
	 mpz_sub(mq_y2,mq_y2,j);
	 mpz_div(mq_y2,mq_y2,q);
	 mpz_invert(mq_y2,mq_y2,q);
	 mpz_mod(mq_y2,mq_y2,q);
	 mpz_mul(mq_y1,mq_y1,mq_y2);
	 mpz_mod(m_q,mq_y1,q);
	 
	 //CRT
	 mpz_t p_q,result;
	 mpz_init(p_q);
	 mpz_init(result);
	 
	 mpz_sub(p_q,m_q,m_p);
	 mpz_mul(p_q,p_q,p_ni);
	 mpz_mod(p_q,p_q,q);
	 mpz_mul(p_q,p_q,p);
	 mpz_add(result,m_p,p_q);
	  
	 finish = clock();
	 printf("time is :%fms\n",(double)(finish - start) * 1000 / CLOCKS_PER_SEC);
	 gmp_printf("解密得到明文m = %Zd\n\n",result);
	 
	 return 0;
}