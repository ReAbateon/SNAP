#include "header.h"

init();

for(int r=0; r<TESTSIZE; r++){
    __asm volatile ("" ::: "memory");
    cyccnt_before = DWT->CYCCNT;
    __asm volatile ("" ::: "memory");
    inference(testset[r]);
    __asm volatile ("" ::: "memory");
    cyccnt_after = DWT->CYCCNT;
    __asm volatile ("" ::: "memory");
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: %d\r\n", r, out);
}