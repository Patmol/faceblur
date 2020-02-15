using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using ImageProcessorCore;
using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;

namespace faceblur
{
    class Program
    {
        private static string SUBSCRIPTION_KEY = Environment.GetEnvironmentVariable("FACE_SUBSCRIPTION_KEY");
        private static string ENDPOINT = Environment.GetEnvironmentVariable("FACE_ENDPOINT");

        static async Task Main(string[] args)
        {
            var client = Program.Authenticate(Program.ENDPOINT, Program.SUBSCRIPTION_KEY);

            Console.WriteLine("Detecting faces ...");
            var facesPosition = await Program.DetectFaceExtract(client, args, RecognitionModel.Recognition02);

            Console.WriteLine("Bluring faces ...");
            Program.BlurImage(facesPosition);
        }

        private static IFaceClient Authenticate(string endpoint, string key)
        {
            return new FaceClient(new ApiKeyServiceClientCredentials(key)) { Endpoint = endpoint };
        }

        private static async Task<IList<(string, IList<Rectangle>)>> DetectFaceExtract(IFaceClient client, string[] paths, string recongnitionModel)
        {                        
            var result = new List<(string, IList<Rectangle>)>();

            foreach (var path in paths)
            {
                IList<DetectedFace> detectedFaces;
                var facesToBlur = new List<Rectangle>();

                using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
                {
                    detectedFaces = await client.Face.DetectWithStreamAsync(stream);

                    Console.WriteLine($"{detectedFaces.Count} faces detected in the image {path}");

                    foreach (var detectedFace in detectedFaces)
                    {
                        facesToBlur.Add(new Rectangle(
                            detectedFace.FaceRectangle.Left,
                            detectedFace.FaceRectangle.Top,
                            detectedFace.FaceRectangle.Width,
                            detectedFace.FaceRectangle.Height));
                    }
                }

                result.Add((path, facesToBlur));
            }

            return result;
        }

        private static void BlurImage(IList<(string, IList<Rectangle>)> imageToBlurs)
        {
            foreach (var imageToBlur in imageToBlurs)
            {
                Console.WriteLine($"Bluring image {imageToBlur.Item1} ...");

                var path = Path.GetDirectoryName(imageToBlur.Item1);
                var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(imageToBlur.Item1);
                var extension = Path.GetExtension(imageToBlur.Item1);
                var blurFile = $"{path}/{fileNameWithoutExtension}-blur{extension}";

                if (File.Exists(blurFile))
                {
                    File.Delete(blurFile);
                }

                using (var stream = File.OpenRead(imageToBlur.Item1))
                using (var output = File.OpenWrite(blurFile))
                {
                    var image = new Image<Color, uint>(stream);
                
                    foreach (var face in imageToBlur.Item2)
                    {
                        image = image.BoxBlur(20, face);
                    }

                    image.SaveAsJpeg(output);
                }

                Console.WriteLine("Image blured");
            }
        }
    }
}