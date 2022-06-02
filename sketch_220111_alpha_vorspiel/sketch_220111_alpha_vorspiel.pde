import oscP5.*;
import netP5.*;

PShape eyeOpen;
PShape eyeClosed;
PImage eye;
PFont font;

OscP5 oscP5;
NetAddress myRemoteLocation;

String currentState;

void setup(){
  pixelDensity(displayDensity());
  size (1000, 1000);
  frameRate(25);
  eye = loadImage("eye_open_small.png");
  imageMode(CENTER);
  font = loadFont("SpaceMono-Bold-110.vlw");
  textFont(font, 110);
  
  oscP5 = new OscP5(this, 12000);
  myRemoteLocation = new NetAddress("127.0.0.1",12000);
}


void draw(){
  
  background(0);
  
  stateOne();
  //stateTwo();
  //stateThree();
  //stateFour();
  //stateFive();
  
 
}

void stateOne(){
  background(0);
  
  pushMatrix();
    translate(width/2, height/9);
    textAlign(CENTER);
    textSize(190);
    text("a.l.p.h.a", 10, height/6);
    textAlign(CENTER);
    textSize(45);
    text("SELECT ONE OR MULTIPLE OF", 10, height/1.5);
    text("THE SURROUNDING OBJECTS", 10, height/1.4);
    text("AND STEP IN FRONT OF THE CAMERA", 10, height/1.3); 
    text("TO INITIALIZE THE SYSTEM", 10, height/1.2); 
  
    eye.resize(150, 0);
    image(eye, 10, height/2.2); 
  popMatrix();
}

void stateTwo(){
   background(0);
  
  pushMatrix();
    translate(width/2, height/9);
    textAlign(CENTER);
    textSize(190);
    text("a.l.p.h.a", 10, height/6);
    textAlign(CENTER);
    textSize(45);
    text("THE SYSTEM IS CAPTURING YOUR DATA", 10, height/2.2);
    text("PLEASE STAND STILL", 10, height/2); 
  popMatrix();
}

void stateThree(){
  background(0);
  
  pushMatrix();
    translate(width/2, height/9);
    textAlign(CENTER);
    textSize(190);
    text("a.l.p.h.a", 10, height/6);
    textAlign(CENTER);
    textSize(45);
    text("GENERATING TEXT", 10, height/2.2);
    text("PLEASE WAIT...", 10, height/2); 
  popMatrix();
}

void stateFour(){
   String teacher = "Marco is a happy man.";

  background(0);
  
  pushMatrix();
    translate(width/2, height/9);
    textAlign(CENTER);
    textSize(190);
    text("a.l.p.h.a", 10, height/6);
    textAlign(CENTER);
    textSize(45);
    text(teacher.substring(0, 10), 10, height/2.2);
    text(teacher.substring(11, teacher.length()), 10, height/2); 
  popMatrix();
}

void stateFive(){
   background(0);
  
  pushMatrix();
    translate(width/2, height/9);
    textAlign(CENTER);
    textSize(190);
    text("a.l.p.h.a", 10, height/6);
    textAlign(CENTER);
    textSize(45);
    text("COLLECT YOUR", 10, height/2.2);
    text("PRINTED POEM", 10, height/2); 
  popMatrix();
}

/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage theOscMessage) {
  /* print the address pattern and the typetag of the received OscMessage */
  //print("### received an osc message.");
  print(theOscMessage.addrPattern());
  print(theOscMessage.get(0));
  //println(" typetag: "+theOscMessage.typetag());
  //currentState = theOscMessage;
}
