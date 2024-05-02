extern "C" {
#include "model_fixed.h"
}

#define MAX_MSG_SIZE 32768

static inline float round_with_mode(float v, round_mode_t round_mode) {
  if (round_mode == ROUND_MODE_FLOOR) {
    return floorf(v);
  } else if (round_mode == ROUND_MODE_NEAREST) {
    return floorf(v + 0.5f);
  } else {
    return v;
  }
}

void setup() {

  // Initialize serial port
  Serial.begin(115200);

  // Initialize pin for blinking LED
  pinMode(PIN_LED, OUTPUT);

  // Wait for initialization
  while (!Serial && millis() < 5000);

  // Notify readyness
  Serial.println("READY");
}

void loop() {
  static unsigned int inference_count = 0;
  static char buf[MAX_MSG_SIZE];
  static float finputs[MODEL_INPUT_DIMS];
  static input_t inputs;
  static output_t outputs;

  // Read message sent by host
  int msg_len = Serial.readBytesUntil('\n', buf, MAX_MSG_SIZE);
  if (msg_len < 1) {
    // Nothing read, send READY again to make sure we got acknowledged and try again at next iteration
    Serial.println("READY");
    return;
  } else if (msg_len != MAX_MSG_SIZE) {
    // Terminator character is discarded from buffer unless number of bytes read equals max length
    // Artificially increment the message length to account for the missing terminator.
    msg_len++;
  }
  Serial.println(msg_len);

  // Convert received string to floats
  char *pbuf = buf;
  int i = 0;
  while ((pbuf - buf) < msg_len && *pbuf != '\r' && *pbuf != '\n') {
    finputs[i] = strtof(pbuf, &pbuf);
    i++;
    pbuf++; // skip delimiter
  }

  //TODO: Convert inputs from floating-point to fixed-point
 float temps;
  for(int i = 0; i < MODEL_INPUT_DIM_0 ; i++){
    for(int k = 0; k<MODEL_INPUT_DIM_1; k++){
      temps = finputs[i * MODEL_INPUT_DIM_1 + k];
      inputs[i][k] = clamp_to(MODEL_INPUT_NUMBER_T, round_with_mode((1 << MODEL_INPUT_SCALE_FACTOR )*temps, MODEL_INPUT_ROUND_MODE));
    }
    
  }
  digitalWrite(PIN_LED, HIGH);
  // Run inference
  cnn(inputs, outputs);
  digitalWrite(PIN_LED, LOW); 

  // Get output class
  unsigned int label = 0;
  float max_val = outputs[0];
  for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
    if (max_val < outputs[i]) {
      max_val = outputs[i];
      label = i;
    }
  }

  inference_count++;

  char msg[64];
  snprintf(msg, sizeof(msg), "%d,%d,%f", inference_count, label, (double)max_val); // force double cast to workaround -Werror=double-promotion since printf uses variadic arguments so promotes to double automatically
  Serial.println(msg);

}
