There are five phases. Only the small draft model is ever trained, and the big target model is never touched.
**1. Collect the teacher's answers (Modal, paid)**  The big model answers about 5,000 training prompts once. Its answers become the "textbook" the small model learns from. We save both the text and the exact tokens.
**2. Train the two adapters (Kaggle, free)**  The small model reads each prompt and practises predicting the big model's answer word by word. Two versions are trained from the same textbook:
*   **Tier 1, the light version:**  only the final output layer is nudged. It is a small correction on top of what the model already predicts.
*   **Tier 2, the heavy version:**  small trainable pieces (LoRA) are added inside every layer. This is the "best we can do with training" reference.
Both are then merged into the model's weights so they add no cost at run time.
**3. Train the "when to stop guessing" rule (Kaggle, free)**  The trained small model replays the training prompts, and we log each guess as accepted or rejected by the big model. A tiny classifier then learns from four confidence signals when the next guess is likely to be rejected, so the small model stops drafting early. This is done separately for Tier 1 and Tier 2.
**4. Tune and decide how much data is needed (Kaggle, free)**
*  All settings (learning rate, number of passes, adapter size, stop threshold) are chosen on the 320 validation prompts only.
*  We train on a quarter, half and all of the base training set. If the score is still climbing at the end, we bring in the extra 3,254 prompts. If it has flattened, we stop.
**5. Freeze, then test (Kaggle free, then Modal paid)**  The chosen models are locked and their fingerprints recorded. Only then is the test set opened: first for acceptance rate on all 400 prompts, then for real speed on 200 prompts with the big model running live.
Tier 0 has no training phase. It only uses lookup caches and a fixed confidence threshold, which is why it is the "free" baseline the trained tiers are compared against.
