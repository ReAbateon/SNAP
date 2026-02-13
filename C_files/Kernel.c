static const uint16_t lane_masks_u16[8][8] = {
  {1,0,0,0,0,0,0,0},
  {0,1,0,0,0,0,0,0},
  {0,0,1,0,0,0,0,0},
  {0,0,0,1,0,0,0,0},
  {0,0,0,0,1,0,0,0},
  {0,0,0,0,0,1,0,0},
  {0,0,0,0,0,0,1,0},
  {0,0,0,0,0,0,0,1},
};

static const uint16x8_t weights = {128, 64, 32, 16, 8, 4, 2, 1};

static inline uint16x8_t v_ones_u16(void)   { return vdupq_n_u16(1); }
static inline uint16x8_t v_zeros_u16(void) { return vdupq_n_u16(0); }

static inline uint16x8_t toggle_lane(uint16x8_t v, int lane){
    uint16x8_t m = vld1q_u16(lane_masks_u16[lane]); // load 16B
    return veorq_u16(v, m);
}

int kernel(int16_t *sample, const int16_t (*thresholds)[8], const uint16_t (*features)[8], const int16_t *outcome, const uint8_t *map, int index){
    uint32_t zero;
    int16_t output;

    int starting_point = map[path_index[index]];
    uint16x8_t path = path_bits[index];

    while (1) {
        int16x8_t  thresh = vld1q_s16(thresholds[starting_point]);
        uint16x8_t feat   = vld1q_u16(features[starting_point]);

        int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, feat);

        mve_pred16_t pred = vcmpleq_s16(samp, thresh);
        uint16x8_t pred_bit = vpselq_u16(v_ones_u16(), v_zeros_u16(), pred);

        mve_pred16_t cmp = vcmpneq_u16(pred_bit, path);

        if ((uint16_t)cmp == 0) {
            output = outcome[starting_point];
            return (int)output;
        }

        zero = __builtin_ctz((uint32_t)(uint16_t)cmp) >> 1;

        path = toggle_lane(path, (int)zero);
        path_bits[index] = path;

        path_index[index] = vmladavaq_u16(0, path, weights);
        starting_point = map[path_index[index]];
    }
}