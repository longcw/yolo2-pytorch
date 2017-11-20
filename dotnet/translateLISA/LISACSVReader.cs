using System.IO;
using System.Collections.Generic;
using Schemas;
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
                index_dictionary.Add(headers[i],i);
            }

        }
        annotation ReadLine()
        {
            string line = reader.ReadLine();
            string[] vals = line.Split(",");
            annotation rtn = new annotation();
            uint index;
            if(!index_dictionary.TryGetValue("Filename",out index))
            {
                rtn.filename=vals[index];
            }
            else
            {
                throw new KeyNotFoundException(
                    "Could not get Filename from line"+
                line);

            }
            if(!index_dictionary.TryGetValue("Filename",out index))
            {
                rtn.filename=vals[index];
            }
            else
            {
                throw new KeyNotFoundException(
                    "Could not get Filename from line"+
                line);

            }

            return rtn;

        }



    }
}