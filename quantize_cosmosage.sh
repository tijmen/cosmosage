version="v14"
dir="/home/tijmen/cosmosage/models/mistral_cosmosage_$version/"
python quant_autogptq.py $dir/lora_out/merged/ $dir/quant4/ wikitext
python quant_autogptq.py $dir/lora_out/merged/ $dir/quant8/ wikitext --bits 8

