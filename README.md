# Tennis ELO model 

Dissertation expanding upon [538's tennis prediction ELO model](https://fivethirtyeight.com/features/serena-williams-and-the-difference-between-all-time-great-and-greatest-of-all-time/), but focusing solely on women's tennis. 

`TLDR:`

- Uses ITF data as well as WTA (source: [Jeff Sackman](https://github.com/JeffSackmann)) but weights ITF games lower.
- Considers game score when deciding player ability change after match.
- Considers a players historical as well as current ability.
- Considers players form in an attempt to preempt upsets.

Full write up: [SOURCE](dissertation.pdf)

## 1 - Setup

Install dependencies: 

```
poetry install
```

Activate environment:

```
poetry shell
```

## 2 - Running

Arguments (all required):
- yf - year from (**WARNING:** don't go pre 2010)
- yt - year to 
- ts - test size (in years)

**Sample:**

Gets data from 2010 to 2020, fits to 2000 -> 2018 and predicts on 2019 and 2020

```
python run --yf 2010 --yt 2020 --ts 2 
```

## 3 - Results 

All model outputs are saved to `data/03_output/`.

These include:
- Model performance (accuracy and brier scores) 
- Rankings (as of end of test data)
- Model calibration
- Predictions appended to test data
