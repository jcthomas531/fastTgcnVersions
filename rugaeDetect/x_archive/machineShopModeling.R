library(MachineShop)
library(dplyr)
library(doSNOW)
registerDoSNOW(makeCluster(12))

#helper function
allZero <- function(x) {
  all(x== 0) 
}


#set seed
seed <- 826
set.seed(seed)

#testing
inFile <- "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/decim8500/norm0.02_desc1.3/combinedDesc.csv"

#XGBModel is just the generic framework for all extreme gradient boosting
#XGBTreeModel is the conventional implementation
# modelinfo("XGBModel")
# modelinfo("XGBTreeModel")
# args(XGBTreeModel)


dat <- read.csv(inFile)
#number of patients
nPats =  length(unique(dat$patId))



#get just the pfh data
datPfh <- dat |>
  select(patId, label, x, y, z, nx, ny, nz, starts_with("pfh")) |>
  #remove any variables that have only 0, this is something we will have to revisit
  select(-where(allZero)) |>
  #paring down data set for faster testing
  #slice_sample(n = 5000) |>
  #making label into a factor
  mutate(label = factor(label, levels = c(0,1)))





#attempting new formula logic to ensure patId does not mess with it
#a vector of everything besides label and patId
varsPfh <- setdiff(names(datPfh), c("label", "patId"))
#make into formula
formPfh <- reformulate(varsPfh, response = "label")

#create model frame
#one thing to check here will be ensuring that patId plays no role in things like dimension reduction
#it shouldnt, but just something to keep in mind, the user guide says it accepts "-" in formula
mf <- ModelFrame(
  formula = formPfh,
  #formula = label ~ x + y + z + nx + ny + nz,
  #formula = label ~ . - patId,
  data = datPfh,
  groups = patId
  )
#attr(mf, "terms")
#formula(mf)

cv <- CVControl(folds = nPats, repeats = 1, seed = seed)

#the tuning process can either be done with TunedModel or TunedInput
#it looks like TunedInput offers more flexibility for varying the preprocessing steps
#which may be useful for comparing performance with different local descriptors
#for now I am gonna use TunedModel as it seems a bit more straightforward
#actually perhaps TunedInput does not have the ability to tune hyperparams and the two should be used in conjunction

#setting up tuning
#i believe that this will use default tuning grids of the specified size
xgbTune <- TunedModel(
  #XGBModel is just the generic framework for all extreme gradient boosting
  #XGBTreeModel is the conventional implementation
  object = XGBTreeModel,
  grid = c(
    nrounds = 5,
    eta = 3,
    gamma = 3,
    max_depth = 3
  )
)
# expand_modelgrid(
#   TunedModel(XGBTreeModel, grid = TuningGrid(3))
# )

#intersection over union metric
iouPoint <- MLMetric(
  function(observed, predicted, ...) {
    #logicals for classification
    pred1 <- predicted == 1
    obs1 <- observed == 1
    #creating metric
    numer <- sum(pred1 & obs1)
    denom <- sum(pred1 | obs1)
    return(numer/denom)
  },
  name = "iouPoint",
  label = "Intersection over union for point classification",
  maximize = TRUE
)

#model specification
#in my previous work there has been a recipe as the input argument here but documentation a model frame will work
xgbSpec <- ModelSpecification(
  input = mf,
  model = xgbTune,
  control = cv,
  metrics = c(
    "iouPoint",
    "roc_auc",
    "auc", 
    "accuracy",
    "roc_index", 
    "fnr", 
    "fpr", 
    "tnr", 
    "tpr", 
    "pr_auc", 
    "precision", 
    "recall",
    "sensitivity",
    "specificity"
  )
)



# metricinfo(datPfh$label) %>% names
# ?auc()

#fitting
xgbFit <- fit(xgbSpec)
#this object should hold both the cross validation information as well as the final fitted model
aaa <- as.MLModel(xgbFit)
ccc <- summary(aaa)
ddd <- ccc$TrainingStep1
#can also navigate the structure this way: bbb <- aaa@steps@.Data[[1]]@log
ddd |>
  filter(selected)

predict(xgbFit)


#how should this be set up in snakemake?
#a different rule for each model or one rule that run all models
#a single rule means there is a lot of overhead if something fails
