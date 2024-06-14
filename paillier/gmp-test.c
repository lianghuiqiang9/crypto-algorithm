#include <stdio.h>
#include <gmp.h>
int main()
{
        mpz_t a, b, c;
        mpz_init(a);
        mpz_init(b);
        mpz_init(c);
        printf("========= Input a and b => Output a + b =========\n");
        //printf("[-] a = ");
        //gmp_scanf("%Zd", a);
        //printf("[-] b = ");
        //gmp_scanf("%Zd", b);
        //mpz_add(c, a, b);
        //gmp_printf("[+] c = %Zd\n",c);

        mpz_init_set_str(a,"12345678900987654321",10);
        mpz_init_set_str(b,"98765432100123456789",10);
        mpz_mul(c, a, b);
        gmp_printf("c = %Zd\n", c);

        mpz_t d;
        mpz_init(d);
        mpz_nextprime(d,a);
        gmp_printf("d = %Zd\n", d);







        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(c);
        return 0;
}
