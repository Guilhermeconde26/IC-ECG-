#include <Chirale_TensorFlowLite.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include "model_data.h" 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

// Pinagem display 
#define TFT_CS    10
#define TFT_DC     9
#define TFT_RST    8
#define TFT_MOSI  11
#define TFT_SCK   12
SPIClass spi = SPIClass(FSPI);
Adafruit_ST7789 tft = Adafruit_ST7789(&spi, TFT_CS, TFT_DC, TFT_RST);

// Pinagem ECG
#define ECG_PIN   1
#define LO_PLUS   4
#define LO_MINUS  5

// GRAFICO 
#define W 320
#define H 240
#define GRAPH_TOP    40
#define GRAPH_BOTTOM 230
#define GRAPH_LEFT   10
#define GRAPH_RIGHT  310

// BPM 
int lastPeakIndex = -1000;
float bpm = 0;

#define RR_MIN_SAMPLES 50 

// AMOSTRAGEM 
#define FS 100
#define DURACAO 10
#define N_AMOSTRAS (FS * DURACAO)

int ecg_buffer[N_AMOSTRAS];
int idx = 0;

// FILTRO PASSA-BAIXA(Temporario)
#define FC 25.0
#define ALPHA (2 * PI * FC / (2 * PI * FC + FS))

float ecg_filt = 2048;

int filtrarECG(int x) {
  ecg_filt = ecg_filt + ALPHA * (x - ecg_filt);
  return (int)ecg_filt;
}

// FEATURES
#define N_INPUTS 11
float features[N_INPUTS];


// REDE NEURAL (TENSORFLOW)
tflite::AllOpsResolver resolver;

constexpr int TENSOR_ARENA_SIZE = 80 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;

TfLiteTensor* input;
TfLiteTensor* output;

// GRAFICO 
int lastX = GRAPH_LEFT;
int lastY = (GRAPH_TOP + GRAPH_BOTTOM) / 2;

// GRID(GRAFICO) 
void drawGrid() {
  tft.fillScreen(ST77XX_BLACK);

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.println("ECG (Testes)");

  tft.drawRect(
    GRAPH_LEFT,
    GRAPH_TOP,
    GRAPH_RIGHT - GRAPH_LEFT,
    GRAPH_BOTTOM - GRAPH_TOP,
    ST77XX_WHITE
  );

  uint16_t gridColor = tft.color565(80, 80, 80);

  for (int y = GRAPH_TOP; y <= GRAPH_BOTTOM; y += 40) {
    tft.drawFastHLine(GRAPH_LEFT, y,
      GRAPH_RIGHT - GRAPH_LEFT, gridColor);
  }

  lastX = GRAPH_LEFT;
}

// Funcao do BPM
void detectarBPM(int signal, int sampleIndex) {

  static int prev = 0;
  static float mean = 2048;

  mean = 0.99 * mean + 0.01 * signal;

  float threshold = mean + 120;  

  bool pico =
    (signal > threshold) &&
    (signal > prev) &&
    (sampleIndex - lastPeakIndex > RR_MIN_SAMPLES);

  if (pico) {

    int rr = sampleIndex - lastPeakIndex;

    if (lastPeakIndex > 0) {
      bpm = 60.0 * FS / rr;
    }

    lastPeakIndex = sampleIndex;
  }

  prev = signal;
}

// Funcao das Features
void calcularFeatures() {

  float mean = 0;
  int minVal = 4095;
  int maxVal = 0;

  for (int i = 0; i < N_AMOSTRAS; i++) {
    int v = ecg_buffer[i];
    mean += v;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }

  mean /= N_AMOSTRAS;

  float variance = 0;
  float energy = 0;
  float absSum = 0;
  int zeroCross = 0;

  for (int i = 0; i < N_AMOSTRAS; i++) {
    float x = ecg_buffer[i] - mean;
    variance += x * x;
    energy += ecg_buffer[i] * ecg_buffer[i];
    absSum += abs(x);

    if (i > 0) {
      if ((ecg_buffer[i] > mean && ecg_buffer[i - 1] < mean) ||
          (ecg_buffer[i] < mean && ecg_buffer[i - 1] > mean)) {
        zeroCross++;
      }
    }
  }

  variance /= N_AMOSTRAS;

  float stdDev = sqrt(variance);
  float rms = sqrt(energy / N_AMOSTRAS);

  features[0]  = mean;
  features[1]  = stdDev;
  features[2]  = minVal;
  features[3]  = maxVal;
  features[4]  = maxVal - minVal;
  features[5]  = rms;
  features[6]  = variance;
  features[7]  = absSum / N_AMOSTRAS;
  features[8]  = zeroCross;
  features[9]  = energy;
  features[10] = (float)zeroCross / N_AMOSTRAS;
}

// Funcao da REDE
void rodarIA() {

  calcularFeatures();

  float input_scale = input->params.scale;
  int input_zero = input->params.zero_point;

  for (int i = 0; i < N_INPUTS; i++) {
    input->data.int8[i] =
      (int8_t)(features[i] / input_scale + input_zero);
  }

  if (interpreter->Invoke() != kTfLiteOk) return;

  int8_t y_int8 = output->data.int8[0];
  float y = (y_int8 + 128) * 0.00390625;

  tft.fillRect(0, 200, 320, 40, ST77XX_BLACK);
  tft.setCursor(10, 200);
  tft.setTextSize(2);

  if (y > 0.5) {
    tft.setTextColor(ST77XX_RED);
    tft.println("ARRITMIA");
  } else {
    tft.setTextColor(ST77XX_GREEN);
    tft.println("NORMAL");
  }

  tft.fillRect(200, 10, 120, 30, ST77XX_BLACK);
  tft.setCursor(200, 10);
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_CYAN);
  tft.print("BPM:");
  tft.print((int)bpm);

}

// Funcao da Correcao
void corrigirEspelhamento(uint8_t madctl) { 
  tft.startWrite(); 
  tft.writeCommand(ST77XX_MADCTL); 
  tft.spiWrite(madctl); 
  tft.endWrite(); 
  }

// SETUP 
void setup() {

  Serial.begin(9600);

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  spi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);

  tft.init(240, 320);
  tft.setRotation(1);
  corrigirEspelhamento(0x28);
  drawGrid();

  model = tflite::GetModel(ecg_model_int8_tflite);

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );

  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

// LOOP
void loop() {

  bool eletrodoSolto =
    digitalRead(LO_PLUS) || digitalRead(LO_MINUS);

  if (eletrodoSolto) {
    idx = 0;
    ecg_filt = 2048;

    tft.fillRect(0, 200, 320, 40, ST77XX_BLACK);
    tft.setCursor(10, 200);
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_YELLOW);
    tft.println("DESCONECTADO");
    bpm = 0;
    lastPeakIndex = -1000;

    delay(200);
    return;
  }

  int raw = analogRead(ECG_PIN);
  int currentValue = filtrarECG(raw);

  detectarBPM(currentValue, idx);

  ecg_buffer[idx++] = currentValue;

  int y = map(currentValue, 0, 4095,
    GRAPH_BOTTOM - 1, GRAPH_TOP + 1);

  int x = lastX + 1;

  if (x >= GRAPH_RIGHT) {
    drawGrid();
    x = GRAPH_LEFT;
  }

  tft.drawLine(lastX, lastY, x, y, ST77XX_GREEN);

  lastX = x;
  lastY = y;

  if (idx >= N_AMOSTRAS) {
    idx = 0;
    rodarIA();
  }

  delay(10);
}
