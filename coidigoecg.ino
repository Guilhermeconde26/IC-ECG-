 #include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>

// Pinagem display 
#define TFT_CS    10
#define TFT_DC     9
#define TFT_RST    8
#define TFT_MOSI  11
#define TFT_SCK   12

// Pinagem display
#define ECG_PIN   1 
#define LO_PLUS   4
#define LO_MINUS  5

// Area do grafico no display
#define W 320 // largura
#define H 240 // Altura
#define GRAPH_TOP    40
#define GRAPH_BOTTOM 230
#define GRAPH_LEFT   10
#define GRAPH_RIGHT  310

SPIClass spi = SPIClass(FSPI);
Adafruit_ST7789 tft = Adafruit_ST7789(&spi, TFT_CS, TFT_DC, TFT_RST);

// Variveis
int lastValue = 2048;
int currentValue = 2048;
int lastX = GRAPH_LEFT;
int lastY = (GRAPH_TOP + GRAPH_BOTTOM) / 2;

void drawGrid() {
  tft.fillScreen(ST77XX_BLACK);

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.println("ECG - Tempo Real");

  tft.drawRect(
    GRAPH_LEFT,
    GRAPH_TOP,
    GRAPH_RIGHT - GRAPH_LEFT,
    GRAPH_BOTTOM - GRAPH_TOP,
    ST77XX_WHITE
  );

  uint16_t gridColor = tft.color565(80, 80, 80);
  for (int y = GRAPH_TOP; y <= GRAPH_BOTTOM; y += 40) {
    tft.drawFastHLine(
      GRAPH_LEFT,
      y,
      GRAPH_RIGHT - GRAPH_LEFT,
      gridColor
    );
  }

  lastX = GRAPH_LEFT;
}
void corrigirEspelhamento(uint8_t madctl) {
  tft.startWrite();
  tft.writeCommand(ST77XX_MADCTL);
  tft.spiWrite(madctl);
  tft.endWrite();
}

void setup() {
  Serial.begin(9600);

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  spi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);

  tft.init(240, 320);
  tft.setRotation(1); 

  corrigirEspelhamento(0x28); 
  drawGrid();
}

void loop() {
  bool eletrodoSolto =
    (digitalRead(LO_PLUS) == 1) ||
    (digitalRead(LO_MINUS) == 1);

  if (!eletrodoSolto) {
    lastValue = currentValue;
    currentValue = analogRead(ECG_PIN);
  } else {
    currentValue = (lastValue + currentValue) / 2;
  }

  Serial.println(currentValue);

  int y = map(
    currentValue,
    0, 4095,
    GRAPH_BOTTOM - 1,
    GRAPH_TOP + 1
  );

  int x = lastX + 1;
  if (x >= GRAPH_RIGHT) {
    drawGrid();
    x = GRAPH_LEFT;
  }

  tft.drawLine(lastX, lastY, x, y, ST77XX_GREEN);

  lastX = x;
  lastY = y;

  delay(10);
}