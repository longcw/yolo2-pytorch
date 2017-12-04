using System;
using System.Xml;
using System.Xml.Serialization;
using System.IO;
using Schemas;
using System.Drawing;
using translateLISA;
using System.Collections.Generic;
namespace transateLISA
{
    class Program
    {
        static void writeImageToFiles(ref annotation currentAnnotation,uint imageNum, ref XmlSerializer ser,
        ref LinkedList<annotationObject> currentObjects)
        {
             Image   im = Image.FromFile(Path.Combine("inputImages",currentAnnotation.filename));
               currentAnnotation.size = new annotationSize();
               currentAnnotation.size.height=(short)im.Height;
               currentAnnotation.size.width=(short)im.Width;
               currentAnnotation.size.depth=3;
               currentAnnotation.@object= new annotationObject[currentObjects.Count];
               currentObjects.CopyTo(currentAnnotation.@object,0);
               currentObjects.Clear();
             string  image_path = imageNum + ".jpg";
               currentAnnotation.filename = image_path;
              string  annotation_path =
               Path.Combine(new string[]{"Annotations","" + imageNum +".xml"});
              FileStream fs = new FileStream(annotation_path,FileMode.Create); 
               ser.Serialize(fs,currentAnnotation);
               im.Save(Path.Combine("JPEGImages",image_path));
        }
        static void Main(string[] args)
        {
            LISACSVReader reader = new LISACSVReader("allFrames.csv");
            XmlSerializer ser = new XmlSerializer(typeof(annotation));
            LinkedList<annotationObject> currentObjects = new LinkedList<annotationObject>();
            annotation currentAnnotation = reader.ReadLine();
            currentObjects.AddLast(currentAnnotation.@object[0]);
           
            
            uint imageNum = 1;
            annotation read;
            while(reader.Peek()>=0){
                read = reader.ReadLine();
                if(read.filename.Equals(currentAnnotation.filename))
                {
                    currentObjects.AddLast(read.@object[0]);
                }
                else
                {
                writeImageToFiles(ref currentAnnotation,imageNum,ref ser,ref currentObjects);
                imageNum++;
                currentAnnotation=read;
                currentObjects.AddLast(currentAnnotation.@object[0]);
                }
            }
            writeImageToFiles(ref currentAnnotation,imageNum,ref ser,ref currentObjects);
        }
    }
}
