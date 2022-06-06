import oscP5.*;
import netP5.*;

PShape eyeOpen;
PShape eyeClosed;
PImage eye;
PFont font;
PFont fontBody;


OscP5 oscP5;
NetAddress myRemoteLocation;

//String currentState;
int currentState;
int textBody = 35;
String prompt;

void setup(){
  pixelDensity(displayDensity());
  //size (1200, 720);
  //Use to make size screen dependent
  fullScreen();
  frameRate(25);
  eye = loadImage("eye_open_small.png");
  imageMode(CENTER);
  font = loadFont("SpaceMono-Bold-190.vlw");
  fontBody = loadFont("SpaceMono-Bold-35.vlw");
  
  oscP5 = new OscP5(this, 12000);
  myRemoteLocation = new NetAddress("127.0.0.1",12000);
}


void draw(){
  
  background(0);
  
  //stateOne();
  //stateTwo();
  //stateThree();
  stateFour();
  //stateFive();
  
  switch(currentState) {
  case 1: 
    //println("Alpha");  // Does not execute
    stateOne();
    break;
  case 2: 
    //println("Bravo");  // Prints "Bravo"
    stateTwo();
    break;
  case 3:
    stateThree();
    break;
  case 4:
    stateFour();
    break;
  case 5:
    stateFive();
    break;
  default:
    //println("Zulu");   // Does not execute
    break;
}
  
 
}

void stateOne(){
  background(0);
  
  pushMatrix();
    translate(width/2, height/2);
    textAlign(CENTER);
    textFont(font);
    textSize(190);
    text("a.l.p.h.a", 0, - (8 * textBody));
    textAlign(CENTER);
    textFont(fontBody);
    textSize(textBody);
    text("SELECT ONE OR MULTIPLE OF", 0, (7 * textBody));
    text("THE SURROUNDING OBJECTS", 0, (8 * textBody));
    text("AND STEP IN FRONT OF THE CAMERA", 0, (9 * textBody)); 
    text("TO INITIALIZE THE SYSTEM", 0, (10 * textBody)); 
  
    eye.resize(150, 0);
    image(eye, 0, 0); 
  popMatrix();
}

void stateTwo(){
   background(0);
  
  pushMatrix();
    translate(width/2, height/2);
    textAlign(CENTER);
    textFont(font);
    textSize(190);
    text("a.l.p.h.a", 0, - (8 * textBody));
    textAlign(CENTER);
    textFont(fontBody);
    textSize(textBody);
    text("THE SYSTEM IS CAPTURING YOUR DATA", 0, (7 * textBody));
    text("PLEASE STAND STILL", 10, (8 * textBody)); 
    
    eye.resize(150, 0);
    image(eye, 0, 0); 
    
  popMatrix();
}

void stateThree(){
  background(0);
  
  pushMatrix();
    translate(width/2, height/2);
    textAlign(CENTER);
    textFont(font);
    textSize(190);
    text("a.l.p.h.a", 0, -(8 * textBody));
    textAlign(CENTER);
    textFont(fontBody);
    textSize(textBody);
    text("GENERATING TEXT", 0, (7 * textBody));
    text("PLEASE WAIT...", 0, (8 * textBody)); 
  popMatrix();
}

void stateFour(){
  //String teacher = "Marco is a happy man.";

  background(0);
  
  pushMatrix();
    translate(width/2, height/2);
    textAlign(CENTER);
    textFont(font);
    textSize(190);
    //text("a.l.p.h.a", 0, height/6);
    textAlign(CENTER);
    textFont(fontBody);
    textSize(textBody);
    //text(prompt.substring(0, 10), 0, height/2.2);
    //text(prompt.substring(11, prompt.length()), 0, height/2); 
    
    float fraction = prompt.length() * 0.10;
   
    //for (int i = 0; i < prompt.length(); i++ ){
      if(prompt.length() <= 75){
        text(prompt, 0, 0);
      }
      else if ((prompt.length() > 75) && (prompt.length() <= 200)){
        text(prompt.substring(0, (int(fraction) * 2)), 0, textBody - (4 * textBody)); 
        text(prompt.substring((int(fraction) * 2), (int(fraction) * 4)), 0, textBody - (3 * textBody)); 
        text(prompt.substring((int(fraction) * 4), (int(fraction) * 6)), 0, textBody - (2 * textBody));
        text(prompt.substring((int(fraction) * 6), (int(fraction) * 8)), 0, (textBody - textBody));
        text(prompt.substring((int(fraction) * 8), prompt.length()), 0, textBody);
      }
      
       else if ((prompt.length() > 200) && (prompt.length() <= 300)){
        text(prompt.substring(0, (int(fraction) * 2)), 0, textBody - (4 * textBody)); 
        text(prompt.substring((int(fraction) * 2), (int(fraction) * 4)), 0, textBody - (3 * textBody)); 
        text(prompt.substring((int(fraction) * 4), (int(fraction) * 6)), 0, textBody - (2 * textBody));
        text(prompt.substring((int(fraction) * 6), (int(fraction) * 8)), 0, (textBody - textBody));
        text(prompt.substring((int(fraction) * 8), prompt.length()), 0, textBody);
      }
      
      
      else if ((prompt.length() > 300) && (prompt.length() <= 400)) {
        //text(prompt.substring(0, int(fraction)), 0, 0);
        text(prompt.substring(0, (int(fraction) * 2)), 0, textBody - (4 * textBody)); 
        text(prompt.substring((int(fraction) * 2), (int(fraction) * 4)), 0, textBody - (3 * textBody)); 
        text(prompt.substring((int(fraction) * 4), (int(fraction) * 6)), 0, textBody - (2 * textBody));
        text(prompt.substring((int(fraction) * 6), (int(fraction) * 8)), 0, (textBody - textBody));
        text(prompt.substring((int(fraction) * 8), prompt.length()), 0, textBody);
      } else {
        
        text(prompt.substring(0, (int(fraction) * 1)), 0, textBody - (9 * textBody)); 
        text(prompt.substring((int(fraction) * 1), (int(fraction) * 2)), 0, textBody - (8 * textBody)); 
        text(prompt.substring((int(fraction) * 2), (int(fraction) * 3)), 0, textBody - (7 * textBody));
        text(prompt.substring((int(fraction) * 3), (int(fraction) * 4)), 0, textBody - (6 * textBody));
        text(prompt.substring((int(fraction) * 4), (int(fraction) * 5)), 0, textBody - (5 * textBody));
        text(prompt.substring((int(fraction) * 5), (int(fraction) * 6)), 0, textBody - (4 * textBody));
        text(prompt.substring((int(fraction) * 6), (int(fraction) * 7)), 0, textBody - (3 * textBody));
        text(prompt.substring((int(fraction) * 7), (int(fraction) * 8)), 0, textBody - (2 * textBody));
        text(prompt.substring((int(fraction) * 8), (int(fraction) * 9)), 0, (textBody - textBody));
        text(prompt.substring((int(fraction) * 9), prompt.length()), 0, textBody);
      }
    //}
    
    
  popMatrix();
}

void stateFive(){
   background(0);
  
  pushMatrix();
    translate(width/2, height/10);
    textAlign(CENTER);
    textFont(font);
    textSize(190);
    text("a.l.p.h.a", 0, height/6);
    textAlign(CENTER);
    textFont(fontBody);
    textSize(textBody);
    text("COLLECT YOUR", 0, height/2.2);
    text("PRINTED POEM", 0, height/2); 
  popMatrix();
}

/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage theOscMessage) {
  /* print the address pattern and the typetag of the received OscMessage */
  //print("### received an osc message.");
  //print(theOscMessage.addrPattern());
  //println(" typetag: "+theOscMessage.typetag());
  
  if (theOscMessage.checkAddrPattern("/alpha/state") == true) {
  currentState = theOscMessage.get(0).intValue();
  println(currentState);
  } else if (theOscMessage.checkAddrPattern("/alpha/prompt") == true) {
    //prompt = theOscMessage.addrPattern();
    //println(prompt);
    //print(theOscMessage.addrPattern());
    println(theOscMessage.get(1).stringValue());
    prompt = (theOscMessage.get(1).stringValue());
    
    //print(prompt.length());
  } else {
    
  }
  
  //currentState = theOscMessage;
}
