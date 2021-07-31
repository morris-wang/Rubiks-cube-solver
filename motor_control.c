#include <Stepper.h>                      // degree180 ->step 105   degree90 -> step 60
// Define number of steps per revolution:
const int stepsPerRevolution = 200;
// Initialize the stepper library on pins 10 through 13:
Stepper R = Stepper(stepsPerRevolution, 8,9,10,11);
Stepper L = Stepper(stepsPerRevolution, 7,6,5,4);
Stepper F = Stepper(stepsPerRevolution, 14,15,16,17);
Stepper B = Stepper(stepsPerRevolution, 26,28,30,32);
Stepper D = Stepper(stepsPerRevolution, 31,33,35,37);
Stepper U = Stepper(stepsPerRevolution, 42,44,46,48);
int speed1 = 40;
int speed2 = 0;
int count=0;
char x;
char sol[100];
void setup() {
  Serial.begin(9600);
  // Set the motor speed (RPMs):
  //myStepper.setSpeed(300);
  R.setSpeed(speed1);
  L.setSpeed(speed1);
  F.setSpeed(speed1);
  B.setSpeed(speed1);
  D.setSpeed(speed1);
  U.setSpeed(speed1);
 /* while(count<2)
  {
  R.step(60);
  delay(500);
  R.step(-60);
  delay(500);
  count=count+1;
  }*/
  
}
void loop() {
    if(Serial.available()){
     x = Serial.read();
    Serial.write(x);
    switch(x){
      case 'R':           //R
          R.step(-52);
          delay(500);
          break;
      case  'r':
          R.step(52);
          delay(500);
          break;
      case  'q':
          R.step(102);
          delay(500);
          break;
      case  'L':          //L
          L.step(-51);
          delay(500);
          break;
      case  'l':
          L.step(51);
          delay(500);
          break;
      case  'a':
          L.step(102);
          delay(500);
          break;
      case 'F':         //F
          F.step(-51);
          delay(500);
          break;
      case  'f':
          F.step(51);
          delay(500);
          break;
      case  'x':
          F.step(102);
          delay(500);
          break;
      case  'B':         //B
          B.step(-51);
          delay(500);
          break;
      case  'b':
          B.step(51);
          delay(500);
          break;
      case  'z':
          B.step(100);
          delay(500);
          break;
      case 'U':           //U
          U.step(-52);
          delay(500);
          break;
      case  'u':
          U.step(52);
          delay(500);
          break;
      case  'w':
          U.step(102);
          delay(500);
          break;
      case 'D':           //D
          D.step(-52);
          delay(500);
          break;
      case  'd':
          D.step(52);
          delay(500);
          break;
      case  's':
          D.step(102);
          delay(500);
          break;
      default: 
              break;
    }
    }
     /*R.step(-50); //90 DEGREE-> 50STEPS  180DEGREE->100
          delay(500);
      L.step(50);
          delay(500);
       F.step(-50);
          delay(500);
      B.step(-50);
          delay(500);*/
       /*U.step(100);*/
         /* delay(500);*/
       /*D.step(-200);
          delay(200);*/
    
}
