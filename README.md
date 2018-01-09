# densenet-efficient-model
## Convert 
In order to obtain the models for the [efficient model](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py) in PyTorch. You need to download the models from originl [Torch models](https://github.com/liuzhuang13/DenseNet). And then convert the torch models to PyTorch models. Then they are converted to efficient models.<br>

```
   python convert_torch.py -m densenet_cosine_264_k48.t7
   python convert_efficient.py
```

Note: You need to change the function in the main function in convert_efficient.py if you want to convert other models.

## Validated
All the models in this table can be converted and the results have been validated.

| Network            |Top-1 error    | Download |
| -------------------|---             | -------- |
|DenseNet-264(k=32) |22.1| [Download(129MB)](https://drive.google.com/file/d/1vWWURpd0kW-41dFXzSEKs-IfGUC8kF-P/view?usp=sharing)|
|DenseNet-232(k=48) |21.2 |[Download(214MB)](https://drive.google.com/file/d/1cXj3Z8VCNnKlgefdXuQaRvyK4dctYWhO/view?usp=sharing)
| DenseNet-cosine-264 (k=32)|21.6 | [DenseNet(129MB)](https://drive.google.com/file/d/15KVHM7n2DUPQSgDqqgiHJQxN6n0m1jTC/view?usp=sharing) |
| DenseNet-cosine-264 (k=48)|20.4 | [DenseNet(280MB)](https://drive.google.com/file/d/1mWQIV07n5DnfFSL4_Mic5a2dddSyTcck/view?usp=sharing) |
