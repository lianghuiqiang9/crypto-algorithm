#include<stdio.h>
#include<gmp.h>
#include<string.h>
#include<time.h>

typedef struct {

    mpz_t n;
    mpz_t p;
    mpz_t q;

}paillier_private_key;

typedef struct {

    mpz_t n;
    mpz_t g; // g = n + 1;

}paillier_public_key;

//做一个结构体, 后续会用到小数上. 
typedef struct{

    mpz_t c;

}paillier_ciphertext;

//这个m应该不需要struct吧, 因为c风格不就是这个样子嘛, 先init在说.
typedef struct{

    mpz_t c;

}paillier_plaintext;

//根据n_length生成pk和sk
void PaillierKeyGeneration(int n_length, paillier_private_key sk, paillier_public_key pk){
    //初始化pk, sk

    //生成随机符合规定的素数

    //生成sk

    //生成pk


}

//根据公钥pk进行加密
paillier_ciphertext Encrypt(paillier_public_key pk, paillier_plaintext m){
    
    paillier_ciphertext c = raw_encrypt(pk,m);

    return c;

}

//raw_encrypt 是为了将来的encode打基础
paillier_ciphertext raw_encrypt(paillier_public_key pk, paillier_plaintext m){
    
    paillier_ciphertext c;

    return c;

}


//根据私钥sk进行解密
paillier_plaintext Decrypt(paillier_private_key sk, paillier_ciphertext c){
    paillier_plaintext m;

    return m;

}

//标量乘法
paillier_ciphertext Mul(paillier_public_key pk, paillier_ciphertext c, mpz_t k){
    //paillier_ciphertext c;

    return c;

}

//标量乘法 plaintext
paillier_ciphertext Mul(paillier_public_key pk, paillier_ciphertext c, paillier_plaintext k){
    //paillier_ciphertext c;

    return c;

}

//加法
paillier_ciphertext Add(paillier_public_key pk, paillier_ciphertext c1, paillier_ciphertext c2){
    paillier_ciphertext c3;

    return c3;

}

//加法 明密
paillier_ciphertext Add(paillier_public_key pk, paillier_ciphertext c, paillier_plaintext m){
    //paillier_ciphertext c;

    return c;

}

//减法
paillier_ciphertext Sub(paillier_public_key pk, paillier_ciphertext c1, paillier_ciphertext c2){
    paillier_ciphertext c3;

    return c3;

}

//减法 明密
paillier_ciphertext Sub(paillier_public_key pk, paillier_ciphertext c, paillier_plaintext m){
    //paillier_ciphertext c;

    return c;

}

//重随机化
void Obfuscate(paillier_public_key pk, paillier_ciphertext c){

}

//destory pk

//destory sk

//destory ciphertext

//destory plaintext


int main(){
    int n_length=2048;
    paillier_private_key sk;
    paillier_public_key pk;
    PaillierKeyGeneration(n_length,sk,pk);



}


