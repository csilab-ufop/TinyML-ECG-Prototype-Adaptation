/* Cortex-M4 entry point for the PSoC 6 deterministic replay application. */
#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"
#include "ecg_head_adaptation_app.h"

int main(void)
{
    cy_rslt_t result = cybsp_init();
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    __enable_irq();
    ecg_head_adaptation_app_run();

    for (;;)
    {
    }
}
