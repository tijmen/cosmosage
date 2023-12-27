## History

 - 2023 Nov 21 - project start
 - 2023 Nov 25 - completed fine tune "cosmosage_v1" based on arxiv papers and physics QA pairs, taking ~10 hours on 1x A6000
 - 2023 Nov 26 - cosmosage_v1: took zephyr, trained on asl x4 + camelai physicsqa at lr=5e-5
 - 2023 Nov 27 - cosmosage_v2: took cosmosage_v1, trained on textbooks at lr=5e-5
 - 2023 Nov 27 - cosmosage_v3: took cosmosage_v2, trained on arxiv_physics_instruct_tune_30k at lr=5e-4
              primary function was to see if maybe my learning rate was far too low before
              the idea will be to compare v2 and v3 on the same questions. If the model "forgets"
              the info from the textbooks then the LR is too high. if it becomes much better at
              QA then things are working.
 - 2023 Nov 28 - cosmosage_v3.1: same but lr=5e-5 (10x lower)
 - 2023 Nov 28 - cosmosage_v3.2: same but lr=1e-3 (2x higher)
              
 - 2023 Nov 28: yi_6b: let's try a different model, starting with this chinese-english base model
       its slightly smaller memory footprint should allow a larger batch size
 - 2023 Nov 30: Yi-6B_textbooks_v1: train on textbooks lr=1e-3
 - 2023 Dec 01: Yi-6B_textbooks_v2: eval loss-clean textbooks, switch to LoRA, switch to AdamW, lr=5e-5
 - 2023 Dec 02: Yi-6B_textbooks_v3: same as v2 but lr=5e-4

 - 2023 Dec 02: MiniChatTextBooks: An attempt to work with the MiniMA package
 - 2023 Dec 03: MiniChat_v2: An attempt to work with the axolotl package, lr=5e-5
 - 2023 Dec 03: MiniChat_v3: v2 had an increasing loss function. I want to see if either weight_decay=0.1
              or lr=5e-5 is too high. I'm most suspicious of weight decay. I'll set weight_decay=0
              and monitor the loss. If it's still rising after an initial decay I'll do v4
              with lr=5e-6
 - 2023 Dec 03: MiniChat_v4: v3 didn't get better by setting the weight_decay to 0. I'm gonna try weight_decay = 0.00001
             and lr=5e-6
 - 2023 Dec 04: MiniChat_v5: Run from the base model with lr=5e-7, no weight decay
 - 2023 Dec 04: MiniChat_v6: Second epoch starting from MiniChat_v5 with lr=5e-6
 - 2023 Dec 04: MiniChat_v7: Third epoch starting from MiniChat_v6 with lr=5e-6	
 - 2023 Dec 05: MiniChat_v8: Three epochs over the ArXiv papers starting from MiniChat_v7
 - 2023 Dec 05: MiniChat_v9: Not making much progress on loss, so doing three more epochs
             starting from MiniChat_v8 at double lr: lr=1e-5
 - 2023 Dec 05: MiniChat_v10: three more epochs from v9
 - 2023 Dec 05: MiniChat_v11: three more epochs from v10
 - 2023 Dec 06: MiniChat_v12: Taking MiniChat_v11 as the base model, I'll now use some QA datasets, starting
                              with a bunch of physics questions. This did not work well. lr=1e-5 was much too
                              high. The validation loss got better and then rapidly worse, while the training loss
                              got very good very fast. On subsequent epochs the training loss lowered further, showing
                              strong overfitting. Evaluating this model by inspecting the text it generates shows
                              that it answers every question with a physics answer. I think I need to lower
                              the learning rate and also sprinkle in another general-purpose dataset.
 - 2023 Dec 07: mistral_cosmosage_v1: let's try fine tuning a base model and then instruction tuning it later. 
                                      Randomly selected 10% of textbook+arxiv data. 
                                      QLoRA, ReLORA 150 steps, lr=1e-4, num_epochs=2
 - 2023 Dec 08: mistral_cosmosage_v2: same but over the whole corpus (~48 hours)
 - 2023 Dec 10: mistral_cosmosage_v3: instruction-tune v2 on SlimOrca-Dedup (~36 hours)
 - 2023 Dec 12: mistral_cosmosage_v4: one more pass through physics_astro_qa
 - 2023 Dec 13: mistral_cosmosage_v5: realizing that lora r=8 was too low. 2 more epochs starting
                                      from v2, but at lora r=128. Also setting lr=1e-4.
 - 2023 Dec 16: mistral_cosmosage_v6: 3 more epochs over the pretraining data
 - 2023 Dec 20: mistral_cosmosage_v7: Fine-tune v6 on all QA pairs + SlimOrca, lr=1e-4. Manually stopped at 2.5 epochs when validation loss started increasing
 - 2023 Dec 23: mistral_cosmosage_v8: another pass over the physics+astro+cosmo QA, lr=3e-5. Set for 5 epochs but will monitor validation loss to decide when to stop
 - 2023 Dec 25: Manual evaluation by interacting with mistral_cosmosage_v8. It's quite impressive and seems to be quite good
                at astronomy, in addition to physics and cosmology. The ArXiV dataset seems to have been partially "understood". The model can provide some answers that it couldn't possibly have known without having read those papers. 
                The primary problem is hallucination. The model makes stuff up all the time. One way to detect hallucinations by hand is by regenerating the answer a few times. The hallucination will change, whereas the knowledge it's confident about doesn't. 
 - 2023 Dec 27: Started generating synthetic data based on the ArXiV papers.