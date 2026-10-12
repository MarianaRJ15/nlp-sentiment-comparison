from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def train(model_name, tokenized, compute_metrics, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
