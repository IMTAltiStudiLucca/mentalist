# Covert channel attack to federated learning

we prove that federated learning systems can be turned into covert channels to implement a stealth communication infrastructure.
The main intuition is that an attacker can injects a bias in the global model by submitting purposely crafted samples.
The effect of the model bias is negligible to the other participants.
Yet, the bias can be measured and used to transmit a single bit through the model.


## Run with docker

```
docker run -it -v $(pwd):/home/fedexp gabrielec/fedexp /bin/bash
```

## Run with docker (OSX)
*With the following command you mount the folder $(pwd) into /home/fedexp/src/tests*
```
 docker run -it -v "$(pwd):/home/fedexp/src/tests" gabrielec/fedexp /bin/bash
```

*Then, when you are inside your docker container, you can run attacks with a command as follows*
```
 python score-attack.py ./simulations/siumulazione_XX.yaml >> ./tests/simulation_XX.out 2>&1 &
```


## Test with docker

*From within the test output folder*

```
docker run -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test-label
docker run -v $(pwd):/home/fedexp/src/tests gabrielec/fedexp-test-score
```
