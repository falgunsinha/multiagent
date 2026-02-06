from pathlib import Path
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')
run_id = 'run-20251220_021934-saemm4ho'
run_dir = wandb_dir / run_id
wandb_file = list(run_dir.glob('*.wandb'))[0]

print(f"Analyzing: {wandb_file}")
print("="*80)

ds = datastore.DataStore()
ds.open_for_scan(str(wandb_file))

record_count = 0
history_count = 0

while record_count < 100:  # Only check first 100 records
    try:
        data = ds.scan_data()
        if data is None:
            break
        
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        
        record_count += 1
        record_type = pb.WhichOneof("record_type")
        
        if record_type == "history":
            history_count += 1
            print(f"\n--- History Record #{history_count} (Record #{record_count}) ---")
            history = pb.history
            print(f"Number of items: {len(history.item)}")
            for i, item in enumerate(history.item[:10]):
                print(f"\nItem {i}:")
                print(f"  key: '{item.key}'")
                print(f"  value_json: '{item.value_json}'")
                if item.nested_key:
                    print(f"  nested_key: {list(item.nested_key)}")

                fields = ['value_int', 'value_float', 'value_str', 'value_bool']
                for field in fields:
                    try:
                        if item.HasField(field):
                            print(f"  {field}: {getattr(item, field)}")
                    except:
                        pass
            
            if history_count >= 3:  # Only show first 3 history records
                break
    
    except Exception as e:
        print(f"Error at record {record_count}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*80}")
print(f"Total records processed: {record_count}")
print(f"History records found: {history_count}")

