#include <immintrin.h>

#include "params.h"
#include "poly.h"
//#include "consts.h"

/*************************************************
* Name:        poly_cbd1
*
* Description: Sample a polynomial deterministically from a random,
*              with output polynomial close to centered binomial distribution
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *buf: pointer to input random
*                                     (of length NTRUPLUS_N/4 bytes)
**************************************************/
void poly_cbd1(poly *r, const unsigned char buf[NTRUPLUS_N/4])
{
    const __m256i mask55 = _mm256_set1_epi32(0x55555555);
    const __m256i mask03 = _mm256_set1_epi32(0x03030303);
    const __m256i mask01 = _mm256_set1_epi32(0x01010101);

    for (size_t i = 0; i < NTRUPLUS_N / 256; i++) {
        __m256i f0, f1;
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i s0, s1, s2, s3, s4, s5, s6, s7;
        __m256i u;

        f0 = _mm256_loadu_si256((const __m256i *)&buf[32*i]);
        f1 = _mm256_loadu_si256((const __m256i *)&buf[32*i + NTRUPLUS_N / 8]);

        t0 = _mm256_and_si256(mask55, f0);
        s0 = _mm256_and_si256(mask55, f1);
        t0 = _mm256_add_epi8(t0, mask55);
        s0 = _mm256_sub_epi8(t0, s0);

        f0 = _mm256_srli_epi16(f0, 1);
        f1 = _mm256_srli_epi16(f1, 1);

        t1 = _mm256_and_si256(mask55, f0);
        s1 = _mm256_and_si256(mask55, f1);
        t1 = _mm256_add_epi8(t1, mask55);
        s1 = _mm256_sub_epi8(t1, s1);

        t0 = _mm256_and_si256(mask03, s0);
        t1 = _mm256_and_si256(mask03, s1);
        t0 = _mm256_sub_epi8(t0, mask01);
        t1 = _mm256_sub_epi8(t1, mask01);

        s0 = _mm256_srli_epi16(s0, 2);
        s1 = _mm256_srli_epi16(s1, 2);

        t2 = _mm256_and_si256(mask03, s0);
        t3 = _mm256_and_si256(mask03, s1);
        t2 = _mm256_sub_epi8(t2, mask01);
        t3 = _mm256_sub_epi8(t3, mask01);

        s0 = _mm256_srli_epi16(s0, 2);
        s1 = _mm256_srli_epi16(s1, 2);

        t4 = _mm256_and_si256(mask03, s0);
        t5 = _mm256_and_si256(mask03, s1);
        t4 = _mm256_sub_epi8(t4, mask01);
        t5 = _mm256_sub_epi8(t5, mask01);

        s0 = _mm256_srli_epi16(s0, 2);
        s1 = _mm256_srli_epi16(s1, 2);

        t6 = _mm256_and_si256(mask03, s0);
        t7 = _mm256_and_si256(mask03, s1);
        t6 = _mm256_sub_epi8(t6, mask01);
        t7 = _mm256_sub_epi8(t7, mask01);

        s0 = _mm256_unpacklo_epi8(t0, t1);
        s1 = _mm256_unpacklo_epi8(t2, t3);
        s2 = _mm256_unpacklo_epi8(t4, t5);
        s3 = _mm256_unpacklo_epi8(t6, t7);
        s4 = _mm256_unpackhi_epi8(t0, t1);
        s5 = _mm256_unpackhi_epi8(t2, t3);
        s6 = _mm256_unpackhi_epi8(t4, t5);
        s7 = _mm256_unpackhi_epi8(t6, t7);

        t0 = _mm256_unpacklo_epi16(s0, s1);
        t1 = _mm256_unpacklo_epi16(s2, s3);
        t2 = _mm256_unpackhi_epi16(s0, s1);
        t3 = _mm256_unpackhi_epi16(s2, s3);
        t4 = _mm256_unpacklo_epi16(s4, s5);
        t5 = _mm256_unpacklo_epi16(s6, s7);
        t6 = _mm256_unpackhi_epi16(s4, s5);
        t7 = _mm256_unpackhi_epi16(s6, s7);

        s0 = _mm256_unpacklo_epi32(t0, t1);
        s1 = _mm256_unpackhi_epi32(t0, t1);
        s2 = _mm256_unpacklo_epi32(t2, t3);
        s3 = _mm256_unpackhi_epi32(t2, t3);
        s4 = _mm256_unpacklo_epi32(t4, t5);
        s5 = _mm256_unpackhi_epi32(t4, t5);
        s6 = _mm256_unpacklo_epi32(t6, t7);
        s7 = _mm256_unpackhi_epi32(t6, t7);

        t0 = _mm256_permute2x128_si256(s0, s1, 0x20);
        t1 = _mm256_permute2x128_si256(s2, s3, 0x20);
        t2 = _mm256_permute2x128_si256(s4, s5, 0x20);
        t3 = _mm256_permute2x128_si256(s6, s7, 0x20);
        t4 = _mm256_permute2x128_si256(s0, s1, 0x31);
        t5 = _mm256_permute2x128_si256(s2, s3, 0x31);
        t6 = _mm256_permute2x128_si256(s4, s5, 0x31);
        t7 = _mm256_permute2x128_si256(s6, s7, 0x31);

        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +   0], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  16], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  32], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t1, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  48], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t2));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  64], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t2, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  80], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t3));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  96], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t3, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 112], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t4));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 128], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t4, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 144], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t5));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 160], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t5, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 176], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t6));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 192], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t6, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 208], u);
        u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t7));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 224], u);
        u = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t7, 1));
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 240], u);
    }
}

/*************************************************
* Name:        poly_sotp
*
* Description: Encode a message deterministically using SOTP and a random,
			   with output polynomial close to centered binomial distribution
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *msg: pointer to input message
*              - const uint8_t *buf: pointer to input random
**************************************************/
void poly_sotp(poly *r, const uint8_t msg[NTRUPLUS_N/8], const uint8_t buf[NTRUPLUS_N/4])
{
    uint8_t tmp[NTRUPLUS_N / 4];

    for(int i = 0; i < NTRUPLUS_N / 8; i++)
    {
         tmp[i] = buf[i]^msg[i];
    }

    for(int i = NTRUPLUS_N / 8; i < NTRUPLUS_N / 4; i++)
    {
         tmp[i] = buf[i];
    }

	poly_cbd1(r, tmp);
}

/*************************************************
* Name:        poly_sotp_inv
*
* Description: Decode a message deterministically using SOTP_INV and a random
*
* Arguments:   - uint8_t *msg: pointer to output message
*              - const poly *a: pointer to iput polynomial
*              - const uint8_t *buf: pointer to input random
*
* Returns 0 (success) or 1 (failure)
**************************************************/
int poly_sotp_inv(uint8_t msg[NTRUPLUS_N/8], const poly *a, const uint8_t buf[NTRUPLUS_N/4])
{
	uint8_t t1, t2, t3;
	uint16_t t4;
	uint32_t r = 0;

	for(size_t i = 0; i < NTRUPLUS_N / 8; i++)
	{
		t1 = buf[i                 ];
		t2 = buf[i + NTRUPLUS_N / 8];
		t3 = 0;

		for(int j = 0; j < 8; j++)
		{
			t4 = t2 & 0x1;
			t4 += a->coeffs[8*i + j];
			r |= t4;
			t4 = (t4 ^ t1) & 0x1;
			t3 ^= (uint8_t)(t4 << j);

			t1 >>= 1;
			t2 >>= 1;
		}

		msg[i] = t3;
	}

	r = r >> 1;
	r = (-(uint32_t)r) >> 31;

	return r;
}

/*
void poly_basemul(poly * __restrict r, const poly * __restrict a, const poly * __restrict b)
{
    const int16_t* zetas_ptr = zetas + 624;
    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256i ymm8, ymm9, ymmA, ymmB, ymmC, ymmD, ymmE, ymmF;
    
    ymm0 = _mm256_load_si256((const __m256i *)_16xq);
    ymmF = _mm256_load_si256((const __m256i *)_16xqinv);

    for (size_t base = 0; base < 768;)
    {
        //load
        ymm1 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 48]); //a[3]
        ymm2 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 16]); //a[1]
        ymm3 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 48]); //a[3]
        ymm4 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 16]); //a[1]

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm5 = _mm256_mullo_epi16(ymm3, ymmE); //a[1]*b[3]
        ymm6 = _mm256_mullo_epi16(ymm4, ymmD); //a[3]*b[1]
        ymm7 = _mm256_mullo_epi16(ymm3, ymmD); //a[3]*b[3]
        ymm8 = _mm256_mulhi_epi16(ymm3, ymm2); //a[1]*b[3]
        ymm9 = _mm256_mulhi_epi16(ymm4, ymm1); //a[3]*b[1]
        ymmA = _mm256_mulhi_epi16(ymm3, ymm1); //a[3]*b[3]

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymm8 = _mm256_sub_epi16(ymm8, ymm5);  //a[1]*b[3]
        ymm9 = _mm256_sub_epi16(ymm9, ymm6);  //a[3]*b[1]
        ymmA = _mm256_sub_epi16(ymmA, ymm7);  //a[3]*b[3]

        //add
        ymm8 = _mm256_add_epi16(ymm8, ymm9);

        //load
        ymm2 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 32]); //a[2]
        ymm4 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 32]); //b[2]

        //premul
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm5 = _mm256_mullo_epi16(ymm4, ymmE); //a[2]*b[2]
        ymm6 = _mm256_mullo_epi16(ymm3, ymmE); //a[2]*b[3]
        ymm7 = _mm256_mullo_epi16(ymm4, ymmD); //a[3]*b[2]
        ymmB = _mm256_mulhi_epi16(ymm4, ymm2); //a[2]*b[2]
        ymmC = _mm256_mulhi_epi16(ymm3, ymm2); //a[2]*b[3]
        ymmD = _mm256_mulhi_epi16(ymm4, ymm1); //a[3]*b[2]

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymmB = _mm256_sub_epi16(ymmB, ymm5);  //a[2]*b[2]
        ymmC = _mm256_sub_epi16(ymmC, ymm6);  //a[2]*b[3]
        ymmD = _mm256_sub_epi16(ymmD, ymm7);  //a[3]*b[2]

        //add
        ymm8 = _mm256_add_epi16(ymm8, ymmB); //c[0]
        ymm9 = _mm256_add_epi16(ymmC, ymmD); //c[1]

        //load zeta
        ymmD = _mm256_load_si256((const __m256i *)(&zetas_ptr[0]));
        ymm1 = _mm256_load_si256((const __m256i *)(&zetas_ptr[16]));

        //mul
        ymm5 = _mm256_mullo_epi16(ymm8, ymmD);
        ymm6 = _mm256_mullo_epi16(ymm9, ymmD);
        ymm7 = _mm256_mullo_epi16(ymmA, ymmD);
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm1);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm1);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm1);

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymm5 = _mm256_sub_epi16(ymm8, ymm5);
        ymm6 = _mm256_sub_epi16(ymm9, ymm6);
        ymm7 = _mm256_sub_epi16(ymmA, ymm7);

        //load
        ymm1 = _mm256_load_si256((const __m256i *)(&a->coeffs[base]));
        ymm3 = _mm256_load_si256((const __m256i *)(&b->coeffs[base]));

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);

        //mul
        ymm8 = _mm256_mullo_epi16(ymm3, ymmD); //a[0]*b[0]
        ymm9 = _mm256_mullo_epi16(ymm4, ymmD); //a[0]*b[2]
        ymmA = _mm256_mullo_epi16(ymm3, ymmE); //a[2]*b[0]
        ymmB = _mm256_mulhi_epi16(ymm3, ymm1); //a[0]*b[0]
        ymmC = _mm256_mulhi_epi16(ymm4, ymm1); //a[0]*b[2]
        ymmE = _mm256_mulhi_epi16(ymm3, ymm2); //a[2]*b[0]

        //reduce
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm0);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm0);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm0);
        ymm8 = _mm256_sub_epi16(ymmB, ymm8); //a[0]*b[0]
        ymm9 = _mm256_sub_epi16(ymmC, ymm9); //a[0]*b[2]
        ymmA = _mm256_sub_epi16(ymmE, ymmA); //a[2]*b[0]

        //add
        ymm5 = _mm256_add_epi16(ymm5, ymm8); //c[0] = c[0]*zeta+a[0]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymm9); //c[2] = c[2]*zeta+a[0]*b[2]+a[2]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymmA); //c[2] = c[2]*zeta+a[0]*b[2]+a[2]*b[0]

        //load
        ymm2 = _mm256_load_si256((const __m256i *)(&a->coeffs[base + 16])); //a[1]
        ymm4 = _mm256_load_si256((const __m256i *)(&b->coeffs[base + 16])); //b[1]

        //premul
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm8 = _mm256_mullo_epi16(ymm4, ymmD); //a[0]*b[1]
        ymm9 = _mm256_mullo_epi16(ymm3, ymmE); //a[1]*b[0]
        ymmA = _mm256_mullo_epi16(ymm4, ymmE); //a[1]*b[1]
        ymmB = _mm256_mulhi_epi16(ymm4, ymm1); //a[0]*b[1]
        ymmC = _mm256_mulhi_epi16(ymm3, ymm2); //a[1]*b[0]
        ymmD = _mm256_mulhi_epi16(ymm4, ymm2); //a[1]*b[1]

        //reduce
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm0);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm0);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm0);
        ymm8 = _mm256_sub_epi16(ymmB, ymm8);  //a[0]*b[1]
        ymm9 = _mm256_sub_epi16(ymmC, ymm9);  //a[1]*b[0]
        ymmA = _mm256_sub_epi16(ymmD, ymmA);  //a[1]*b[1]

        //add
        ymm6 = _mm256_add_epi16(ymm6, ymm8); //c[1] = c[1]*zeta+a[0]*b[1]
        ymm6 = _mm256_add_epi16(ymm6, ymm9); //c[1] = c[1]*zeta+a[0]*b[1]+a[1]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymmA); //c[2] = c[2]*zeta+a[0]*b[2]+a[1]*b[1]+a[2]*b[0]

        //store
        _mm256_store_si256((__m256i *)&r->coeffs[base +  0], ymm5);
        _mm256_store_si256((__m256i *)&r->coeffs[base + 16], ymm6);
        _mm256_store_si256((__m256i *)&r->coeffs[base + 32], ymm7);

        //load
        ymm5 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 32]); //a[2]
        ymm6 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 48]); //a[3]
        ymm7 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 32]); //b[2]
        ymm8 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 48]); //b[3]

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);
        ymmB = _mm256_mullo_epi16(ymm5, ymmF);
        ymmC = _mm256_mullo_epi16(ymm6, ymmF);

        //mul
        ymmD = _mm256_mullo_epi16(ymm8, ymmD); //a[0]*b[3]
        ymmE = _mm256_mullo_epi16(ymm7, ymmE); //a[1]*b[2]
        ymmB = _mm256_mullo_epi16(ymm4, ymmB); //a[2]*b[1]
        ymmC = _mm256_mullo_epi16(ymm3, ymmC); //a[3]*b[0]
        ymm1 = _mm256_mulhi_epi16(ymm8, ymm1); //a[0]*b[3]
        ymm2 = _mm256_mulhi_epi16(ymm7, ymm2); //a[1]*b[2]
        ymm5 = _mm256_mulhi_epi16(ymm4, ymm5); //a[2]*b[1]
        ymm6 = _mm256_mulhi_epi16(ymm3, ymm6); //a[3]*b[0]

        //reduce
        ymmD = _mm256_mulhi_epi16(ymmD, ymm0);
        ymmE = _mm256_mulhi_epi16(ymmE, ymm0);
        ymmB = _mm256_mulhi_epi16(ymmB, ymm0);
        ymmC = _mm256_mulhi_epi16(ymmC, ymm0);
        ymmD = _mm256_sub_epi16(ymm1, ymmD); //a[0]*b[3]
        ymmE = _mm256_sub_epi16(ymm2, ymmE); //a[2]*b[1]
        ymmB = _mm256_sub_epi16(ymm5, ymmB); //a[0]*b[3]
        ymmC = _mm256_sub_epi16(ymm6, ymmC); //a[2]*b[1]

        //add
        ymm1 = _mm256_add_epi16(ymmD, ymmE);
        ymm2 = _mm256_add_epi16(ymmB, ymmC);
        ymm1 = _mm256_add_epi16(ymm1, ymm2);

        //store
        _mm256_store_si256((__m256i *)&r->coeffs[base + 48], ymm1);

        base += 64;

        //load
        ymm1 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 48]); //a[3]
        ymm2 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 16]); //a[1]
        ymm3 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 48]); //a[3]
        ymm4 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 16]); //a[1]

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm5 = _mm256_mullo_epi16(ymm3, ymmE); //a[1]*b[3]
        ymm6 = _mm256_mullo_epi16(ymm4, ymmD); //a[3]*b[1]
        ymm7 = _mm256_mullo_epi16(ymm3, ymmD); //a[3]*b[3]
        ymm8 = _mm256_mulhi_epi16(ymm3, ymm2); //a[1]*b[3]
        ymm9 = _mm256_mulhi_epi16(ymm4, ymm1); //a[3]*b[1]
        ymmA = _mm256_mulhi_epi16(ymm3, ymm1); //a[3]*b[3]

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymm8 = _mm256_sub_epi16(ymm8, ymm5);  //a[1]*b[3]
        ymm9 = _mm256_sub_epi16(ymm9, ymm6);  //a[3]*b[1]
        ymmA = _mm256_sub_epi16(ymmA, ymm7);  //a[3]*b[3]

        //add
        ymm8 = _mm256_add_epi16(ymm8, ymm9);

        //load
        ymm2 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 32]); //a[2]
        ymm4 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 32]); //b[2]

        //premul
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm5 = _mm256_mullo_epi16(ymm4, ymmE); //a[2]*b[2]
        ymm6 = _mm256_mullo_epi16(ymm3, ymmE); //a[2]*b[3]
        ymm7 = _mm256_mullo_epi16(ymm4, ymmD); //a[3]*b[2]
        ymmB = _mm256_mulhi_epi16(ymm4, ymm2); //a[2]*b[2]
        ymmC = _mm256_mulhi_epi16(ymm3, ymm2); //a[2]*b[3]
        ymmD = _mm256_mulhi_epi16(ymm4, ymm1); //a[3]*b[2]

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymmB = _mm256_sub_epi16(ymmB, ymm5);  //a[2]*b[2]
        ymmC = _mm256_sub_epi16(ymmC, ymm6);  //a[2]*b[3]
        ymmD = _mm256_sub_epi16(ymmD, ymm7);  //a[3]*b[2]

        //add
        ymm8 = _mm256_add_epi16(ymm8, ymmB); //c[0]
        ymm9 = _mm256_add_epi16(ymmC, ymmD); //c[1]

        //load zeta
        ymmD = _mm256_load_si256((const __m256i *)(&zetas_ptr[0]));
        ymm1 = _mm256_load_si256((const __m256i *)(&zetas_ptr[16]));

        //mul
        ymm5 = _mm256_mullo_epi16(ymm8, ymmD);
        ymm6 = _mm256_mullo_epi16(ymm9, ymmD);
        ymm7 = _mm256_mullo_epi16(ymmA, ymmD);
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm1);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm1);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm1);

        //reduce
        ymm5 = _mm256_mulhi_epi16(ymm5, ymm0);
        ymm6 = _mm256_mulhi_epi16(ymm6, ymm0);
        ymm7 = _mm256_mulhi_epi16(ymm7, ymm0);
        ymm5 = _mm256_sub_epi16(ymm5, ymm8);
        ymm6 = _mm256_sub_epi16(ymm6, ymm9);
        ymm7 = _mm256_sub_epi16(ymm7, ymmA);

        //load
        ymm1 = _mm256_load_si256((const __m256i *)(&a->coeffs[base])); //a[0]
        ymm3 = _mm256_load_si256((const __m256i *)(&b->coeffs[base])); //b[0]

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);

        //mul
        ymm8 = _mm256_mullo_epi16(ymm3, ymmD); //a[0]*b[0]
        ymm9 = _mm256_mullo_epi16(ymm4, ymmD); //a[0]*b[2]
        ymmA = _mm256_mullo_epi16(ymm3, ymmE); //a[2]*b[0]
        ymmB = _mm256_mulhi_epi16(ymm3, ymm1); //a[0]*b[0]
        ymmC = _mm256_mulhi_epi16(ymm4, ymm1); //a[0]*b[2]
        ymmE = _mm256_mulhi_epi16(ymm3, ymm2); //a[2]*b[0]

        //reduce
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm0);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm0);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm0);
        ymm8 = _mm256_sub_epi16(ymmB, ymm8); //a[0]*b[0]
        ymm9 = _mm256_sub_epi16(ymmC, ymm9); //a[0]*b[2]
        ymmA = _mm256_sub_epi16(ymmE, ymmA); //a[2]*b[0]

        //add
        ymm5 = _mm256_add_epi16(ymm5, ymm8); //c[0] = c[0]*zeta+a[0]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymm9); //c[2] = c[2]*zeta+a[0]*b[2]+a[2]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymmA); //c[2] = c[2]*zeta+a[0]*b[2]+a[2]*b[0]

        //load
        ymm2 = _mm256_load_si256((const __m256i *)(&a->coeffs[base + 16])); //a[1]
        ymm4 = _mm256_load_si256((const __m256i *)(&b->coeffs[base + 16])); //b[1]

        //premul
        ymmE = _mm256_mullo_epi16(ymm2, ymmF);

        //mul
        ymm8 = _mm256_mullo_epi16(ymm4, ymmD); //a[0]*b[1]
        ymm9 = _mm256_mullo_epi16(ymm3, ymmE); //a[1]*b[0]
        ymmA = _mm256_mullo_epi16(ymm4, ymmE); //a[1]*b[1]
        ymmB = _mm256_mulhi_epi16(ymm4, ymm1); //a[0]*b[1]
        ymmC = _mm256_mulhi_epi16(ymm3, ymm2); //a[1]*b[0]
        ymmD = _mm256_mulhi_epi16(ymm4, ymm2); //a[1]*b[1]

        //reduce
        ymm8 = _mm256_mulhi_epi16(ymm8, ymm0);
        ymm9 = _mm256_mulhi_epi16(ymm9, ymm0);
        ymmA = _mm256_mulhi_epi16(ymmA, ymm0);
        ymm8 = _mm256_sub_epi16(ymmB, ymm8);  //a[0]*b[1]
        ymm9 = _mm256_sub_epi16(ymmC, ymm9);  //a[1]*b[0]
        ymmA = _mm256_sub_epi16(ymmD, ymmA);  //a[1]*b[1]

        //add
        ymm6 = _mm256_add_epi16(ymm6, ymm8); //c[1] = c[1]*zeta+a[0]*b[1]
        ymm6 = _mm256_add_epi16(ymm6, ymm9); //c[1] = c[1]*zeta+a[0]*b[1]+a[1]*b[0]
        ymm7 = _mm256_add_epi16(ymm7, ymmA); //c[2] = c[2]*zeta+a[0]*b[2]+a[1]*b[1]+a[2]*b[0]

        //store
        _mm256_store_si256((__m256i *)&r->coeffs[base +  0], ymm5);
        _mm256_store_si256((__m256i *)&r->coeffs[base + 16], ymm6);
        _mm256_store_si256((__m256i *)&r->coeffs[base + 32], ymm7);

        //load
        ymm5 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 32]); //a[2]
        ymm6 = _mm256_load_si256((const __m256i *)&a->coeffs[base + 48]); //a[3]
        ymm7 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 32]); //b[2]
        ymm8 = _mm256_load_si256((const __m256i *)&b->coeffs[base + 48]); //b[3]

        //premul
        ymmD = _mm256_mullo_epi16(ymm1, ymmF);
        ymmB = _mm256_mullo_epi16(ymm5, ymmF);
        ymmC = _mm256_mullo_epi16(ymm6, ymmF);

        //mul
        ymmD = _mm256_mullo_epi16(ymm8, ymmD); //a[0]*b[3]
        ymmE = _mm256_mullo_epi16(ymm7, ymmE); //a[1]*b[2]
        ymmB = _mm256_mullo_epi16(ymm4, ymmB); //a[2]*b[1]
        ymmC = _mm256_mullo_epi16(ymm3, ymmC); //a[3]*b[0]
        ymm1 = _mm256_mulhi_epi16(ymm8, ymm1); //a[0]*b[3]
        ymm2 = _mm256_mulhi_epi16(ymm7, ymm2); //a[1]*b[2]
        ymm5 = _mm256_mulhi_epi16(ymm4, ymm5); //a[2]*b[1]
        ymm6 = _mm256_mulhi_epi16(ymm3, ymm6); //a[3]*b[0]

        //reduce
        ymmD = _mm256_mulhi_epi16(ymmD, ymm0);
        ymmE = _mm256_mulhi_epi16(ymmE, ymm0);
        ymmB = _mm256_mulhi_epi16(ymmB, ymm0);
        ymmC = _mm256_mulhi_epi16(ymmC, ymm0);
        ymmD = _mm256_sub_epi16(ymm1, ymmD); //a[0]*b[3]
        ymmE = _mm256_sub_epi16(ymm2, ymmE); //a[2]*b[1]
        ymmB = _mm256_sub_epi16(ymm5, ymmB); //a[0]*b[3]
        ymmC = _mm256_sub_epi16(ymm6, ymmC); //a[2]*b[1]

        //add
        ymm1 = _mm256_add_epi16(ymmD, ymmE);
        ymm2 = _mm256_add_epi16(ymmB, ymmC);
        ymm1 = _mm256_add_epi16(ymm1, ymm2);

        //store
        _mm256_store_si256((__m256i *)&r->coeffs[base + 48], ymm1);

        base += 64;        
        zetas_ptr += 32;
    }

    return;
}
*/