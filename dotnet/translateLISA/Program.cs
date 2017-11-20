using System;
using System.Xml;
using System.Xml.Serialization;
using System.IO;
using Schemas;
namespace transateLISA
{
    class Program
    {
        static void Main(string[] args)
        {
            XmlSerializer ser = new XmlSerializer(typeof(annotation));
            annotation blargh = new annotation();
            blargh.filename="test_output.jpg";
            blargh.folder="dat_data";
            blargh.owner = new annotationOwner();
            blargh.owner.name="Trent";
            blargh.owner.flickrid="Beep Bop Boolean Boogie";
            blargh.segmented=1;
            /* */
            blargh.size=new annotationSize(); 
            blargh.size.depth=3;
            blargh.size.height=101;
            blargh.size.width=1337;
            
            blargh.source=new annotationSource();
            blargh.source.annotation = "wtf is this?";
            blargh.source.database="Dat db.";
            blargh.source.flickrid=7;
            blargh.source.image="Edgy";

            blargh.@object = new annotationObject[2];
            Console.WriteLine(blargh.@object.Length);
            blargh.@object[0]=new annotationObject();

            blargh.@object[0].bndbox=new annotationObjectBndbox();
            blargh.@object[0].bndbox.xmin=1;
            blargh.@object[0].bndbox.ymin=2;
            blargh.@object[0].bndbox.xmax=3;
            blargh.@object[0].bndbox.ymax=4;
            blargh.@object[0].difficult=1;
            blargh.@object[0].name="asdf";
            blargh.@object[0].pose="FORWARD!!!";
            blargh.@object[0].truncated=0;
            
            FileStream out_fs = new FileStream("test_output.xml",FileMode.Create);
            ser.Serialize(out_fs,blargh);
        }
    }
}
