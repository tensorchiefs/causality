df = NULL

load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPGLinear_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Lin', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPGLinear_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Lin', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPG2x3+x_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Cubic', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPG2x3+x_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Cubic', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPG0.5exp_ModelLS_E500.RData")
df = rbind(df, data.frame(DGP = 'Exp', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run1/triangle_mixed_DPG0.5exp_ModelCS_E500.RData")
df = rbind(df, data.frame(DGP = 'Exp', model="CS", val_NLL=mean(val_loss[491:500])))


df_small = NULL

load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPGLinear_ModelLS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Lin', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPGLinear_ModelCS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Lin', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPG2x3+x_ModelLS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Cubic', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPG2x3+x_ModelCS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Cubic', model="CS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)

load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPG0.5exp_ModelLS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Exp', model="LS", val_NLL=mean(val_loss[491:500])))
rm(val_loss)
load("summerof24/runs/triangle_structured_mixed/run_small_net/triangle_mixed_DPG0.5exp_ModelCS_E500.RData")
df_small = rbind(df_small, data.frame(DGP = 'Exp', model="CS", val_NLL=mean(val_loss[491:500])))

df
df_small
