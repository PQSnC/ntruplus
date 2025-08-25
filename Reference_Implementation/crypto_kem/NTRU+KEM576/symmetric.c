#include "symmetric.h"
#include "fips202.h"

void hash_g(uint8_t *buf, const uint8_t *msg)
{
	uint8_t data[1 + NTRUPLUS_POLYBYTES] = {0x1};

	for (int i = 0; i < NTRUPLUS_POLYBYTES; i++)
	{
		data[i+1] = msg[i];
	}
	
	shake256(buf,NTRUPLUS_N/4,data,NTRUPLUS_POLYBYTES+1);
}

void hash_h_kem(uint8_t *buf, const uint8_t *msg)
{
	uint8_t data[1 + NTRUPLUS_N/8 + NTRUPLUS_SYMBYTES + 1] = {0x2};

	for (int i = 0; i < NTRUPLUS_N/8 + NTRUPLUS_SYMBYTES + 1; i++)
	{
		data[i+1] = msg[i];
	}

	shake256(buf,NTRUPLUS_SSBYTES + NTRUPLUS_N/4,data,NTRUPLUS_N/8 + NTRUPLUS_SYMBYTES+2);
}
