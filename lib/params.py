
# Parameter setting of neurons from Lee
paramSin = [{'V_th':-55.0, 'V_reset':-70.0}, {'V_th':-55.0, 'V_reset':-70.0}, {'V_th':-55.0, 'V_reset':-70.0}, {'V_th':-55.0, 'V_reset':-70.0}]
paramHom = [{'C_m':2e2, 'V_reset':-67.4, 'E_L':-70.0, 'g_L':20.3, 'a':2.0, 'b':4.0, 'Delta_T':2.0, 'tau_w':120.0, 'V_th':-41.5, 'E_ex':0.0, 'E_in':-85.0, 'tau_syn_ex':0.22, 'tau_syn_in':5.0, 't_ref':2.0},
{'C_m':2e2, 'V_reset':-66.4, 'E_L':-70.0, 'g_L':77.1, 'V_th':-41.6,  'E_ex':0.0, 'E_in':-85.0, 'tau_syn_ex':0.2, 'tau_syn_in':5.5, 't_ref':2.0},
{'C_m':2e2, 'V_reset':-59.9, 'E_L':-70.0, 'g_L':21.4, 'a':2.0, 'b':4.0, 'Delta_T':2.0, 'tau_w':120.0, 'V_th':-41.8, 'E_ex':0.0, 'E_in':-85.0, 'tau_syn_ex':0.29, 'tau_syn_in':9.1, 't_ref':2.0},
{'C_m':2e2, 'V_reset':-65.7, 'E_L':-70.0, 'g_L':26.6, 'a':-1.0, 'b':19.0, 'Delta_T':2.0, 'tau_w':120.0, 'V_th':-43.7, 'E_ex':0.0, 'E_in':-85.0, 'tau_syn_ex':0.28, 'tau_syn_in':12.2, 't_ref':2.0}]

# Connectivity settting
conProb = [ [0.1, 0.6, 0.55, 0.01],
            [0.45, 0.5, 0.6, 0.01],
            [0.35, 0.01, 0.5, 0.5],
            [0.1, 0.01, 0.45, 0.5]]
