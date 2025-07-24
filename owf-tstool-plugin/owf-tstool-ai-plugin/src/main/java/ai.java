import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.*;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.types.DataType;


import java.nio.file.Paths;

public class ai {
    public static void main(String[] args) throws Exception {
        int seq_len = 10;
        int weather_features = 12;

        try (Model model = Model.newInstance("water_level_model")) {
            model.load(Paths.get("src/main/resources")); // Path needs to be a Path object

            Translator<NDList, NDArray> translator = new Translator<NDList, NDArray>() {
                @Override
                public NDArray processOutput(TranslatorContext ctx, NDList list) {
                    return list.singletonOrThrow();
                }

                @Override
                public NDList processInput(TranslatorContext ctx, NDList input) {
                    return input;
                }

                @Override
                public Batchifier getBatchifier() {
                    return null;
                }
            };

            try (NDManager manager = NDManager.newBaseManager();
                 Predictor<NDList, NDArray> predictor = model.newPredictor(translator)) {

                NDArray historical = manager.randomNormal(0, 1, new Shape(1, seq_len, weather_features + 1), DataType.FLOAT32);
                NDArray futureWeather = manager.randomNormal(0, 1, new Shape(1, seq_len, weather_features), DataType.FLOAT32);

                NDList input = new NDList(historical, futureWeather);
                NDArray output = predictor.predict(input);

                System.out.println(output);
            }
        }
    }
}
