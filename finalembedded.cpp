

#include "FastMCP3008.h"

FastMCP3008 adc;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc.init();
}

#define DO_CALC_AVG

void loop() {
  const unsigned long C = 3;
  int curValue;
  unsigned int valMin = 1023;
  unsigned int valMax = 0;
  Serial.print("<");
  for(unsigned long x=0; x<C; ++x) {
    curValue = adc.read(adc.getChannelMask(x));
    if (curValue < 10){
      Serial.print("0");
      Serial.print("0");
      Serial.print("0");
      Serial.print(curValue);
    }
    else if (curValue < 100){
      Serial.print("0");
      Serial.print("0");
      Serial.print(curValue);
    }
    else if (curValue < 1000){
      Serial.print("0");
      Serial.print(curValue);
    }
    if (x < C - 1) {
      Serial.print(" ");
    }
  }
  Serial.print(">");
  Serial.flush();
}