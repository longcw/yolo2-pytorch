using System.IO;
using System.Collections.Generic;
using Schemas;
using System;
using System.Drawing;
namespace translateLISA
{
    public class LISACSVReader
    {
        private StreamReader reader;
        string[] headers;

        Dictionary<string,uint> index_dictionary;
        public LISACSVReader(string filename)
        {
            reader = new StreamReader(new FileStream(filename,FileMode.Open));
            headers = reader.ReadLine().Split(",");
            index_dictionary = new Dictionary<string, uint>(headers.Length);
            for(uint i = 0; i < headers.Length; i++)
            {
                Console.WriteLine("Adding dictionary entry for: " + headers[i] + ": " + i);
                index_dictionary.Add(headers[i],i);
            }

        }
        public int Peek()
        {
            return reader.Peek();
        }
        public annotation ReadLine()
        {
            string line = reader.ReadLine();
            string[] vals = line.Split(",");
            annotation rtn = new annotation();
            uint index;
            index_dictionary.TryGetValue("Filename",out index);
            rtn.filename=vals[index];
            rtn.@object=new annotationObject[1];
            rtn.@object[0] = new annotationObject();
            rtn.@object[0].pose="FORWARD!!!";
            rtn.@object[0].truncated=0;
            rtn.@object[0].bndbox=new annotationObjectBndbox();
            
            index_dictionary.TryGetValue("Annotation tag",out index);
            rtn.@object[0].name=vals[index];


            //Upper left corner X	Upper left corner Y	Lower right corner X	Lower right corner Y	Occluded

            index_dictionary.TryGetValue("Upper left corner X",out index);
            rtn.@object[0].bndbox.xmin=Int16.Parse(vals[index]);

            index_dictionary.TryGetValue("Upper left corner Y",out index);
            rtn.@object[0].bndbox.ymin=Int16.Parse(vals[index]);

            index_dictionary.TryGetValue("Lower right corner X",out index);
            rtn.@object[0].bndbox.xmax=Int16.Parse(vals[index]);
      
            index_dictionary.TryGetValue("Lower right corner Y",out index);
            rtn.@object[0].bndbox.ymax=Int16.Parse(vals[index]);

            index_dictionary.TryGetValue("Occluded",out index);
            rtn.@object[0].difficult=SByte.Parse(vals[index]);

            rtn.folder="LISA";
            rtn.owner=new annotationOwner();
            rtn.owner.flickrid="asdf";
            rtn.owner.name="jkl;";
            rtn.segmented=0;
            rtn.source=new annotationSource();
            rtn.source.annotation = "Beep Bop Boolean Boogie.";
            
            return rtn;

        }



    }
}