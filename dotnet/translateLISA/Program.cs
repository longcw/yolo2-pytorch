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
        static void Main(string[] args)
        {
            LISACSVReader reader = new LISACSVReader("allFrames.csv");
            XmlSerializer ser = new XmlSerializer(typeof(annotation));
            LinkedList<annotationObject> currentObjects = new LinkedList<annotationObject>();
            annotation currentAnnotation = reader.ReadLine();
            currentObjects.AddLast(currentAnnotation.@object[0]);
            FileStream fs;
            
            
            uint imageNum = 1;
            do{
            annotation read = reader.ReadLine();
            if(read.filename.Equals(currentAnnotation.filename))
            {
                currentObjects.AddLast(read.@object[0]);
            }
            else
            {
               Image im = new Bitmap(Path.Combine("JPEGImages",currentAnnotation.filename));
               currentAnnotation.size = new annotationSize();
               currentAnnotation.size.height=(short)im.Height;
               currentAnnotation.size.width=(short)im.Width;
               currentAnnotation.@object= new annotationObject[currentObjects.Count];
               currentObjects.CopyTo(currentAnnotation.@object,0);
               currentObjects.Clear();
               string outpath =
               Path.Combine(new string[]{"Annotations",""+imageNum+".xml"});
               fs = new FileStream(outpath,FileMode.CreateNew); 
               ser.Serialize(fs,currentAnnotation);
               currentAnnotation=read;
               currentObjects.AddLast(currentAnnotation.@object[0]);
            }


            
            }while(reader.Peek()>=0 && imageNum<4);
            
        }
    }
}
