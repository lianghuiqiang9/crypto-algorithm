
import random
import time 
from phe import paillier
# pip install phe

def main():
    l=3072  #1024 #2048 #3072
    size_a=10
    print("n_length : ", l)
    start_time = time.time()
    public_key, private_key = paillier.generate_paillier_keypair(None,l)
    
    end_time = time.time()
    print('keygen     time cost:', (end_time-start_time), 'seconds')

    a = []
    for i in range(size_a):
        x = random.randint(a=1, b=9999999999999999999999)
        a.append(x)
    #print("a : ", a)
    b = random.randint(a=1, b=999)
    #print("b : ", b)

    start_time = time.time()
    cipher_a = [public_key.encrypt(x) for x in a]
    end_time = time.time()
    print('encrypt    time cost:', (end_time-start_time)/size_a, 'seconds')

    #start_time = time.time()
    #[x.obfuscate() for x in cipher_a]
    #end_time = time.time()
    #print('encrypt time    cost:', (end_time-start_time)/size_a*1000*1000, 'microseconds')

    start_time = time.time()
    plain_a=[]
    for x in cipher_a:
        plain_a.append(private_key.decrypt(x))
    end_time = time.time()
    print('decrypt    time cost:', (end_time-start_time)/size_a, 'seconds')
    #print(plain_a)

    start_time = time.time()
    cipher_a_mul_b= [x * b for x in cipher_a]
    end_time = time.time()
    print('E(a)*b     time cost:', (end_time-start_time)/size_a, 'seconds')

    plain_a_mul_b=[]
    for x in cipher_a_mul_b:
        plain_a_mul_b.append(private_key.decrypt(x))
    end_time = time.time()
    #print(plain_a_mul_b)

    start_time = time.time()
    cipher_a_add_b= [x + b for x in cipher_a]
    end_time = time.time()
    print('E(a)+b     time cost:', (end_time-start_time)/size_a, 'seconds')

    plain_a_add_b=[]
    for x in cipher_a_add_b:
        plain_a_add_b.append(private_key.decrypt(x))
    end_time = time.time()
    #print(plain_a_add_b)

    cipher_b = public_key.encrypt(b)
    start_time = time.time()
    cipher_a_add_Eb= [x + cipher_b for x in cipher_a]
    end_time = time.time()
    print('E(a)+E(b)  time cost:', (end_time-start_time)/size_a, 'seconds')

    plain_a_add_Eb=[]
    for x in cipher_a_add_Eb:
        plain_a_add_Eb.append(private_key.decrypt(x))
    #end_time = time.time()
    #print(plain_a_add_Eb)


if __name__=="__main__":
    main()