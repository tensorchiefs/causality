df = NULL

### For 10'000 test samples
# > df
#     DGP model   val_NLL
# 1   Lin    LS 0.7305777
# 2   Lin    CS 0.7167840
# 3 Cubic    LS 0.7398601
# 4 Cubic    CS 0.7267842
# 5   Exp    LS 0.7311983
# 6   Exp    CS 0.7266931

load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPGLinear_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Lin', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPGLinear_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Lin', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPG2x3+x_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Cubic', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPG2x3+x_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Cubic', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPG0.5exp_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Exp', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_continous/run_nodes25/triangle_mixed_DPG0.5exp_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Exp', model="CS", val_NLL=mean(val_loss[491:500])))

df