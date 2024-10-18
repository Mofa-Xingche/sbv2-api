from sbv2_bindings import TTSModel


def main():
    print("Loading models...")
    model = TTSModel.from_path("../models/deberta.onnx", "../models/tokenizer.json")
    print("Models loaded!")

    model.load_sbv2file_from_path("amitaro", "../models/amitaro.sbv2")
    print("All setup is done!")

    with open("output.wav", "wb") as f:
        f.write(
            model.synthesize("おはようございます。", "amitaro", 0, 0.0, 0.5)
        )


if __name__ == "__main__":
    main()
