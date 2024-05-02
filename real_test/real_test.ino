#include <I2S.h>

extern "C" {
#include "model_fixed.h"
}

#define I2S_SAMPLE_RATE 16000  // [16000, 48000] supported by the microphone
#define I2S_BITS_PER_SAMPLE 32 // Data is sent in 32-bit packets over I2S but only 18 bits are used by the microphone, remaining least significant bits are set to 0

static input_t inputs; 
static volatile size_t sample_i = 0;
static output_t outputs;
static volatile boolean ready_for_inference = false;

void processI2SData(uint8_t *data, size_t size) {
    int32_t *data32 = (int32_t*)data;

    // Copy first channel into model inputs
    size_t i = 0;
    for (i = 0; i < size / 8 && sample_i + i < MODEL_INPUT_DIM_0; i++, sample_i++) {
      inputs[sample_i][0] = data32[i * 2] >> 14; // Drop 32 - 18 = 14 unused bits
    }

    if (sample_i >= MODEL_INPUT_DIM_0) {
      ready_for_inference = true;
    }
}

void onI2SReceive() {
  size_t size = I2S.available();
  static uint8_t data[I2S_BUFFER_SIZE];

  if (size > 0) {
    I2S.read(data, size);
    processI2SData(data, size);
  }
}

void setup() {
  Serial.begin(115200);

  // For RFThing-DKAIoT
  pinMode(PIN_LED, OUTPUT);
  pinMode(LS_GPS_ENABLE, OUTPUT);
  digitalWrite(LS_GPS_ENABLE, LOW);
  pinMode(LS_GPS_V_BCKP, OUTPUT);
  digitalWrite(LS_GPS_V_BCKP, LOW);
  pinMode(SD_ON_OFF, OUTPUT);
  digitalWrite(SD_ON_OFF, HIGH);

  delay(100); // Wait for peripheral power rail to stabilize after setting SD_ON_OFF

  // start I2S
  if (!I2S.begin(I2S_PHILIPS_MODE, I2S_SAMPLE_RATE, I2S_BITS_PER_SAMPLE, false)) {
    Serial.println("Failed to initialize I2S!");
    while (1); // do nothing
  }

  I2S.onReceive(onI2SReceive);

  // Trigger a read to start DMA
  I2S.peek();

  Serial.println("READY");
}

bool collectingData = false;

void loop() {
  long long t_start = 0;
  if (!collectingData) {
    unsigned long start_time = millis(); 
    
    
    while (millis() - start_time < 10000) { 
      Serial.println("Collecting Data...");
    }

    collectingData = true;
  }
  
  if (ready_for_inference) {


    // Turn LED on during preprocessing/prediction
    digitalWrite(PIN_LED, HIGH);

    // Start timer
    t_start = millis();

    // Compute DC offset
    int32_t dc_offset = 0;

    for (size_t i = 0; i < sample_i; i++) { // Accumulate samples
      dc_offset += inputs[i][0];
    }

    dc_offset = dc_offset / (int32_t)sample_i; // Compute average over samples

    // Filtering
    for (size_t i = 0; i < sample_i; i++) {
      // Remove DC offset
      inputs[i][0] -= dc_offset;

      // Amplify
      inputs[i][0] = inputs[i][0] << 2;
    }

    // Predict
    cnn(inputs, outputs);

    // Get output class
    unsigned int label = 0;
    float max_val = outputs[0];
    for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
      if (max_val < outputs[i]) {
        max_val = outputs[i];
        label = i;
      }
    }
    
    static char msg[32];
    snprintf(msg, sizeof(msg), "%d,%f,%d", label, (double)max_val, (int)(millis() - t_start));
    Serial.println(msg);
    
    // Turn LED off after prediction has been sent
    digitalWrite(PIN_LED, LOW);
    
    ready_for_inference = false;
    sample_i = 0;
    collectingData = false; // Reset the flag to start collecting data again
  }
}
