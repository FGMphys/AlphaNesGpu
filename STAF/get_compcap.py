import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if not gpus:
    print("Nessuna GPU trovata")
else:
    details = tf.config.experimental.get_device_details(gpus[0])
    cc = details.get("compute_capability")

    if cc is None:
        print("Compute capability non disponibile")
    else:
        major, minor = cc

        sm = f"sm_{major}{minor}"
        print(sm)

