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
/*
void poly_cbd1(poly *r, const unsigned char buf[NTRUPLUS_N/4])
{
    const __m256i ymm0 = _mm256_set1_epi32(0x55555555);
    const __m256i ymm1 = _mm256_set1_epi32(0x03030303);
    const __m256i ymm2 = _mm256_set1_epi32(0x01010101);

    for (size_t i = 0; i < NTRUPLUS_N / 256; i++) {
        __m256i ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymma, ymmb, ymmc, ymmd, ymme;
        __m128i xmmb, xmmc, xmmd, xmme;

        ymm3 = _mm256_loadu_si256((const __m256i *)&buf[32*i]);
        ymm4 = _mm256_loadu_si256((const __m256i *)&buf[32*i + NTRUPLUS_N / 8]);
        ymm5 = _mm256_srli_epi16(ymm3, 1);
        ymm6 = _mm256_srli_epi16(ymm4, 1);

        ymm3 = _mm256_and_si256(ymm0, ymm3);
        ymm4 = _mm256_and_si256(ymm0, ymm4);
        ymm5 = _mm256_and_si256(ymm0, ymm5);
        ymm6 = _mm256_and_si256(ymm0, ymm6);

        ymm3 = _mm256_add_epi8(ymm3, ymm0);
        ymm5 = _mm256_add_epi8(ymm5, ymm0);

        ymm7 = _mm256_sub_epi8(ymm3, ymm4);
        ymm8 = _mm256_sub_epi8(ymm5, ymm6);
        ymm9 = _mm256_srli_epi16(ymm7, 2);
        ymma = _mm256_srli_epi16(ymm8, 2);

        ymm3 = _mm256_and_si256(ymm1, ymm7);
        ymm4 = _mm256_and_si256(ymm1, ymm8);
        ymm5 = _mm256_and_si256(ymm1, ymm9);
        ymm6 = _mm256_and_si256(ymm1, ymma);

        ymm3 = _mm256_sub_epi8(ymm3, ymm2);
        ymm4 = _mm256_sub_epi8(ymm4, ymm2);
        ymm5 = _mm256_sub_epi8(ymm5, ymm2);
        ymm6 = _mm256_sub_epi8(ymm6, ymm2);

        ymm7 = _mm256_srli_epi16(ymm7, 4);
        ymm8 = _mm256_srli_epi16(ymm8, 4);
        ymm9 = _mm256_srli_epi16(ymm9, 4);
        ymma = _mm256_srli_epi16(ymma, 4);

        ymm7 = _mm256_and_si256(ymm1, ymm7);
        ymm8 = _mm256_and_si256(ymm1, ymm8);
        ymm9 = _mm256_and_si256(ymm1, ymm9);
        ymma = _mm256_and_si256(ymm1, ymma);

        ymm7 = _mm256_sub_epi8(ymm7, ymm2);
        ymm8 = _mm256_sub_epi8(ymm8, ymm2);
        ymm9 = _mm256_sub_epi8(ymm9, ymm2);
        ymma = _mm256_sub_epi8(ymma, ymm2);

        ymmb = _mm256_unpacklo_epi8(ymm3, ymm4);
        ymmc = _mm256_unpacklo_epi8(ymm5, ymm6);
        ymmd = _mm256_unpacklo_epi8(ymm7, ymm8);
        ymme = _mm256_unpacklo_epi8(ymm9, ymma);
        ymm3 = _mm256_unpackhi_epi8(ymm3, ymm4);
        ymm4 = _mm256_unpackhi_epi8(ymm5, ymm6);
        ymm5 = _mm256_unpackhi_epi8(ymm7, ymm8);
        ymm6 = _mm256_unpackhi_epi8(ymm9, ymma);

        ymm7 = _mm256_unpacklo_epi16(ymmb, ymmc);
        ymm8 = _mm256_unpacklo_epi16(ymmd, ymme);
        ymm9 = _mm256_unpackhi_epi16(ymmb, ymmc);
        ymma = _mm256_unpackhi_epi16(ymmd, ymme);
        ymmb = _mm256_unpacklo_epi16(ymm3, ymm4);
        ymmc = _mm256_unpacklo_epi16(ymm5, ymm6);
        ymmd = _mm256_unpackhi_epi16(ymm3, ymm4);
        ymme = _mm256_unpackhi_epi16(ymm5, ymm6);

        ymm3 = _mm256_unpacklo_epi32(ymmb, ymmc);
        ymm4 = _mm256_unpackhi_epi32(ymmb, ymmc);
        ymm5 = _mm256_unpacklo_epi32(ymmd, ymme);
        ymm6 = _mm256_unpackhi_epi32(ymmd, ymme);
        ymmb = _mm256_unpacklo_epi32(ymm7, ymm8);
        ymmc = _mm256_unpackhi_epi32(ymm7, ymm8);
        ymmd = _mm256_unpacklo_epi32(ymm9, ymma);
        ymme = _mm256_unpackhi_epi32(ymm9, ymma);

        ymm7 = _mm256_permute2x128_si256(ymm3, ymm4, 0x20);
        ymm8 = _mm256_permute2x128_si256(ymm5, ymm6, 0x20);
        ymm9 = _mm256_permute2x128_si256(ymm3, ymm4, 0x31);
        ymma = _mm256_permute2x128_si256(ymm5, ymm6, 0x31);
        ymm3 = _mm256_permute2x128_si256(ymmb, ymmc, 0x20);
        ymm4 = _mm256_permute2x128_si256(ymmd, ymme, 0x20);
        ymm5 = _mm256_permute2x128_si256(ymmb, ymmc, 0x31);
        ymm6 = _mm256_permute2x128_si256(ymmd, ymme, 0x31);

        xmmb = _mm256_castsi256_si128(ymm3);
        xmmc = _mm256_castsi256_si128(ymm4);
        xmmd = _mm256_castsi256_si128(ymm5);
        xmme = _mm256_castsi256_si128(ymm6);

        ymmb = _mm256_cvtepi8_epi16(xmmb);
        ymmc = _mm256_cvtepi8_epi16(xmmc);
        ymmd = _mm256_cvtepi8_epi16(xmmd);
        ymme = _mm256_cvtepi8_epi16(xmme);

        _mm256_store_si256((__m256i *)&r->coeffs[256*i +   0], ymmb);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  32], ymmc);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 128], ymmd);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 160], ymme);        

        xmmb = _mm256_castsi256_si128(ymm7);
        xmmc = _mm256_castsi256_si128(ymm8);
        xmmd = _mm256_castsi256_si128(ymm9);
        xmme = _mm256_castsi256_si128(ymma);

        ymmb = _mm256_cvtepi8_epi16(xmmb);
        ymmc = _mm256_cvtepi8_epi16(xmmc);
        ymmd = _mm256_cvtepi8_epi16(xmmd);
        ymme = _mm256_cvtepi8_epi16(xmme);

        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  64], ymmb);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  96], ymmc);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 192], ymmd);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 224], ymme);    



        xmmb = _mm256_extracti128_si256(ymm3, 1);
        xmmc = _mm256_extracti128_si256(ymm4, 1);
        xmmd = _mm256_extracti128_si256(ymm5, 1);
        xmme = _mm256_extracti128_si256(ymm6, 1);

        ymmb = _mm256_cvtepi8_epi16(xmmb);
        ymmc = _mm256_cvtepi8_epi16(xmmc);
        ymmd = _mm256_cvtepi8_epi16(xmmd);
        ymme = _mm256_cvtepi8_epi16(xmme);

        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  16], ymmb);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  48], ymmc);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 144], ymmd);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 176], ymme); 


        xmmb = _mm256_extracti128_si256(ymm7, 1);
        xmmc = _mm256_extracti128_si256(ymm8, 1);
        xmmd = _mm256_extracti128_si256(ymm9, 1);
        xmme = _mm256_extracti128_si256(ymma, 1);

        ymmb = _mm256_cvtepi8_epi16(xmmb);
        ymmc = _mm256_cvtepi8_epi16(xmmc);
        ymmd = _mm256_cvtepi8_epi16(xmmd);
        ymme = _mm256_cvtepi8_epi16(xmme);

        _mm256_store_si256((__m256i *)&r->coeffs[256*i +  80], ymmb);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 112], ymmc);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 208], ymmd);
        _mm256_store_si256((__m256i *)&r->coeffs[256*i + 240], ymme);         
    }
}
*/
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
/*
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
*/
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
    const __m256i ymm0 = _mm256_set1_epi8((char)0x55);
    const __m256i ymm1 = _mm256_set1_epi8((char)0xff);
    const __m256i ymm2 = _mm256_set1_epi8((char)0x01);
    const __m256i mask1 = _mm256_set1_epi16(0x00ff);
    const __m256i mask2 = _mm256_set1_epi16(0xff00);
          __m256i ymmf = _mm256_set1_epi8((char)0xff);

    __m256i ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymma, ymmb, ymmc, ymmd, ymme;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i a0, a1, a2, a3, a4, a5, a6, a7;


    for (size_t i = 0; i < NTRUPLUS_N / 256; i++) 
    {
        
        ymm7 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i +  0]);
        ymm8 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 128]);
        ymm9 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 16]);
        ymma = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 144]);

        ymmb = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 64]);
        ymmc = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 192]);    
        ymmd = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 80]);
        ymme = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 208]);

        ymm3 = _mm256_packs_epi16(ymm7, ymmb);      
        ymm4 = _mm256_packs_epi16(ymm8, ymmc);
        ymm5 = _mm256_packs_epi16(ymm9, ymmd);
        ymm6 = _mm256_packs_epi16(ymma, ymme);

        ymm7 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 32]);
        ymm8 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 160]);
        ymm9 = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 48]);
        ymma = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 176]);

        ymmb = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 96]);
        ymmc = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 224]);
        ymmd = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 112]);
        ymme = _mm256_load_si256((const __m256i *)&a->coeffs[256*i + 240]);

        ymm7 = _mm256_packs_epi16(ymm7, ymmb);
        ymm8 = _mm256_packs_epi16(ymm8, ymmc);
        ymm9 = _mm256_packs_epi16(ymm9, ymmd);
        ymma = _mm256_packs_epi16(ymma, ymme);

        ymmb = _mm256_permute2x128_si256(ymm3, ymm4, 0x20);
        ymmc = _mm256_permute2x128_si256(ymm3, ymm4, 0x31);
        ymmd = _mm256_permute2x128_si256(ymm5, ymm6, 0x20);
        ymme = _mm256_permute2x128_si256(ymm5, ymm6, 0x31);
        ymm3 = _mm256_permute2x128_si256(ymm7, ymm8, 0x20);
        ymm4 = _mm256_permute2x128_si256(ymm7, ymm8, 0x31);
        ymm5 = _mm256_permute2x128_si256(ymm9, ymma, 0x20);
        ymm6 = _mm256_permute2x128_si256(ymm9, ymma, 0x31);

        ymm7 = _mm256_slli_epi64(ymm3, 32);
        ymm8 = _mm256_srli_epi64(ymmb, 32);
        ymm9 = _mm256_slli_epi64(ymm4, 32);
        ymma = _mm256_srli_epi64(ymmc, 32);

        ymm7 = _mm256_blend_epi32(ymmb, ymm7, 0xaa);
        ymm8 = _mm256_blend_epi32(ymm3, ymm8, 0x55);
        ymm9 = _mm256_blend_epi32(ymmc, ymm9, 0xaa);
        ymma = _mm256_blend_epi32(ymm4, ymma, 0x55);

        ymmb = _mm256_slli_epi64(ymm5, 32);
        ymmc = _mm256_srli_epi64(ymmd, 32);
        ymm3 = _mm256_slli_epi64(ymm6, 32);
        ymm4 = _mm256_srli_epi64(ymme, 32);

        ymmb = _mm256_blend_epi32(ymmd, ymmb, 0xaa);
        ymmc = _mm256_blend_epi32(ymm5, ymmc, 0x55);
        ymmd = _mm256_blend_epi32(ymme, ymm3, 0xaa);
        ymme = _mm256_blend_epi32(ymm6, ymm4, 0x55);

        ymm3 = _mm256_slli_epi32(ymmb, 16);
        ymm4 = _mm256_srli_epi32(ymm7, 16);
        ymm5 = _mm256_slli_epi32(ymmc, 16);
        ymm6 = _mm256_srli_epi32(ymm8, 16);

        ymm3 = _mm256_blend_epi16(ymm7, ymm3, 0xaa);
        ymm4 = _mm256_blend_epi16(ymmb, ymm4, 0x55);
        ymm5 = _mm256_blend_epi16(ymm8, ymm5, 0xaa);
        ymm6 = _mm256_blend_epi16(ymmc, ymm6, 0x55);

        ymm7 = _mm256_slli_epi32(ymmd, 16);
        ymm8 = _mm256_srli_epi32(ymm9, 16);
        ymmb = _mm256_slli_epi32(ymme, 16);
        ymmc = _mm256_srli_epi32(ymma, 16);

        ymm7 = _mm256_blend_epi16(ymm9, ymm7, 0xaa);
        ymm8 = _mm256_blend_epi16(ymmd, ymm8, 0x55);
        ymm9 = _mm256_blend_epi16(ymma, ymmb, 0xaa);
        ymma = _mm256_blend_epi16(ymme, ymmc, 0x55);

        ymmb = _mm256_and_si256(ymm3, mask1);
        ymmc = _mm256_and_si256(ymm4, mask1);
        ymmd = _mm256_and_si256(ymm7, mask2);
        ymme = _mm256_and_si256(ymm8, mask2);
        ymm3 = _mm256_srli_epi16(ymm3, 8);
        ymm4 = _mm256_srli_epi16(ymm4, 8);
        ymm7 = _mm256_slli_epi16(ymm7, 8);
        ymm8 = _mm256_slli_epi16(ymm8, 8);

        a0 = _mm256_xor_si256(ymmb, ymm7);
        a1 = _mm256_xor_si256(ymm3, ymmd);
        a2 = _mm256_xor_si256(ymmc, ymm8);
        a3 = _mm256_xor_si256(ymm4, ymme);

        t0 = _mm256_and_si256(ymm5, mask1);
        t1 = _mm256_and_si256(ymm6, mask1);
        t2 = _mm256_slli_epi16(ymm9, 8);
        t3 = _mm256_slli_epi16(ymma, 8);

        a4 = _mm256_xor_si256(t0, t2);
        a6 = _mm256_xor_si256(t1, t3);

        t0 = _mm256_srli_epi16(ymm5, 8);
        t1 = _mm256_srli_epi16(ymm6, 8);
        t2 = _mm256_and_si256(ymm9, mask2);
        t3 = _mm256_and_si256(ymma, mask2);

        a5 = _mm256_xor_si256(t0, t2);
        a7 = _mm256_xor_si256(t1, t3);

        ymm7 = a0;
        ymm8 = a1;
        ymm9 = a2;
        ymma = a3;
        ymmb = a4;
        ymmc = a5;
        ymmd = a6;
        ymme = a7;

        ymm3 = _mm256_add_epi8(ymm7, ymm2);
        ymm4 = _mm256_add_epi8(ymm8, ymm2);
        ymm5 = _mm256_add_epi8(ymm9, ymm2);
        ymm6 = _mm256_add_epi8(ymma, ymm2);
        ymm7 = _mm256_add_epi8(ymmb, ymm2);
        ymm8 = _mm256_add_epi8(ymmc, ymm2);
        ymm9 = _mm256_add_epi8(ymmd, ymm2);
        ymma = _mm256_add_epi8(ymme, ymm2);

        ymm5 = _mm256_slli_epi16(ymm5, 2);
        ymm6 = _mm256_slli_epi16(ymm6, 2);
        ymm7 = _mm256_slli_epi16(ymm7, 4);
        ymm8 = _mm256_slli_epi16(ymm8, 4);
        ymm9 = _mm256_slli_epi16(ymm9, 6);
        ymma = _mm256_slli_epi16(ymma, 6);

        ymm5 = _mm256_xor_si256(ymm3, ymm5);
        ymm6 = _mm256_xor_si256(ymm4, ymm6);
        ymm7 = _mm256_xor_si256(ymm7, ymm9);
        ymm8 = _mm256_xor_si256(ymm8, ymma);

        ymm3 = _mm256_xor_si256(ymm5, ymm7);
        ymm4 = _mm256_xor_si256(ymm6, ymm8);

        ymm5 = _mm256_loadu_si256((const __m256i *)&buf[32*i]);
        ymm6 = _mm256_loadu_si256((const __m256i *)&buf[32*i + NTRUPLUS_N / 8]);

        ymm7 = _mm256_srli_epi16(ymm6, 1);

        ymm6 = _mm256_and_si256(ymm6, ymm0);
        ymm7 = _mm256_and_si256(ymm7, ymm0);

        ymm3 = _mm256_add_epi8(ymm3, ymm6);
        ymm4 = _mm256_add_epi8(ymm4, ymm7);

        //handling error
        ymm6 = _mm256_srli_epi16(ymm3, 1);
        ymm7 = _mm256_srli_epi16(ymm4, 1);

        ymm6 = _mm256_xor_si256(ymm3, ymm6);
        ymm7 = _mm256_xor_si256(ymm4, ymm7);

        ymm6 = _mm256_and_si256(ymm6, ymm7);
        ymmf = _mm256_and_si256(ymmf, ymm6);

        //extract bits
        ymm3 = _mm256_and_si256(ymm3, ymm0);
        ymm4 = _mm256_and_si256(ymm4, ymm0);
        ymm4 = _mm256_slli_epi16(ymm4, 1);

        ymm3 = _mm256_xor_si256(ymm3, ymm4);
        ymm3 = _mm256_xor_si256(ymm3, ymm1);
        
        ymm3 = _mm256_xor_si256(ymm3, ymm5);

        _mm256_storeu_si256((__m256i *)&msg[32*i], ymm3);
    }

    ymmf = _mm256_xor_si256(ymmf, ymm1);
    ymmf = _mm256_and_si256(ymmf, ymm0);

    return !_mm256_testz_si256(ymmf, ymmf);
}
/*
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
*/
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