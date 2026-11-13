NOTES for Milestone 1
=========================


<style>
.todo {
    display: inline-block; 
    vertical-align: middle;
     margin-right: 12px; 
     background: #f0eadc; 
     color: #380303; 
     font-family: monospace; 
     font-size: 1em;      
     padding: 8px 16px; 
     border-radius: 20px; 
     box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
</style>




## Review of things to do

1. Choose dataset         🆗
    - overview steps:
        - inspect 
            - count samples & target
            - example records
            - distributions
            - missing data
            - inconsistent data
            -  possible ambigous or overlapping data
        - Visualize
            - samples per label
            - text stats
            - duplicates, etc
            - word frequency plot/word cloud


    - <span class="todo"> graded questions </span>:
        1. *long text*: ----  describe the dataset ✅
            - which data 
            - content.. samples
            - task decided to do
            - potential business use
        1. *long text*: ----  initial observation ✅
            - imblanaced?
            - patterns
            - anomalies
            - redundancies
            - topic overlap
        1. *long text*: ----  challenges ✅
            - affecting model performance?
        1. *long text*: ----  planning what to do ✅
        1. *long text*: ----  why this dataset? ✅


2. Problem framing       🆗
    - steps:
        - diagnose challenges 
            - class imabalance
            - lenght variances
            - noise
            - leakage
            - overlapping labels
            - processing constraint/time
        - map what to do
        - explore metrics
            - primary
            - secondary (precission-recall, confusion)
            - protocol, seed
    
    - <span class="todo"> graded questions </span>:
        1. *long text*: ----  one sentence of research question ✅
        1. *long text*: ----  input vs output ✅
        1. *long text*: ----  research question on issues ✅
        1. *long text*: ----  in depth tehcniuqes to use for working on the issues ✅
        1. *long text*: ----  specify metrics  ✅

3. Data prep
    - steps:
        - load
        - cleaning
            - concat headline + summary
        - standardize
            - tokenize
            - max length
            - padding
        - process issues
        - reproducible split
        - documenting pipeline

    - <span class="todo"> graded questions </span>:
        1. *long text*: ----  describe the load & clean steps
        1. *long text*: ----  describe the preprocessing pipeline
        1. *long text*: ----  describe splitting & leakage prevention
        1. *long text*: ----  report counts, imbalance %, weighting, oversampling


4. First model
    - steps:
        - make baseline model
        - compile
        - train & validate
        - visualize

    - <span class="todo"> graded questions </span>:
        1. *long text*: ----  describe model architecture
        1. *long text*: ----  describe training - validation curve
        1. *long text*: ----  report validation metric
        1. *long text*: ----  reflection

5. Report GAIA



        
<div style="display: flex; 
     background: #fee500; 
     border-radius: 12px; 
     box-shadow: 0 2px 10px rgba(0,0,0,0.15);"  >   

---------------------

</div>




# EDA

|  | count | percentage          |
| :------------ | :-----------: | ------------: |
| Total size    |   200,853     |   |
| Missing |
|   - headline  | 6         | 0.003% |
|   - desc      | 19,712      |   9.814%    | 
|   - authors   | 36,620    |   18.2%   |
||||
| Combo missing
| --- no headline   | 1 ||
| --- headline only | 19,707    | 9.812% |
| --- both missing  | 5||


### some comments
- the exact duplicate rows is 13, but havent investigate more
- the repeated link is 41... i wonder if this is sign of true dups
- there are 1,504 dups headlines... but
    - there are several weekly topics such as
        - `sunday roundup`, `top 20 funniest tweets from women this week`, `weekly roundup`
    - about 309 have this multiple date, so.. possibly could cause heavier weight.. not sure if this is delete or adjust weight.
    - the true headline + same date, is the one easy to delete.


# EDA on target

In total, there are 41 labels for 'category' of the news. 

The lowest count is for Education with only 1004 (0.50%). 
The category with most articles is Politics with 16.30%, almost double the 2nd most with only 8.88%. 



## some issues, but not found
- low count label
- strange label

### but there is ambigous ones

ie., `FIFTY`, `GREEN`


## some other issues

**repetitive** 
- `arts`, `arts & cultures`, `cultures & arts` 
    - ➕ -- ok to merge
- `Worldpost`, `THE worldpost`, `world news` 
    - ➕ -- ok to merge
- `parents`, `parenting`
    - ➕ -- ok to merge
-  `style`, `style & beauty`
    - ➕ -- ok to merge

**prob could be repetitive**
- `green` & `environment`
    - ➕ -- ok to merge
- `divorce` & `wedding`
    - ❌ -- seems still unique labels till the end
- `taste` & `food & drink`
    - ➕ -- ok to merge

**maybe overlap?**
- `money` vs `business`
    - QUESTIONABLE
- `healthy living` vs `wellness`
    - ➕ -- ok to merge
- `science` vs `tech`
    - QUESTIONABLE
- `college` vs `education`
    - QUESTIONABLE

**wait,..why is there no topics...**
- `medical` / `health`
    - BUT, there is `fifty` .. which is for people >50yo... and mostly about menopause(?)


## other bad issue

- when combining the headline & content, I found that 
some articles have exact H&C, even same date...,
but different category label!!!!!!
    - There are some that have different dates;
    this makes me think that the editor adjusted the category 
    after the publication, to a more fitting category. 
    So I am thinking that for duplicates, I should use the last date.
