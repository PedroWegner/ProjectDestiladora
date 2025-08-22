# x_range = np.linspace(0.00001, 0.99999, 500) # Gerando o space_plot da fracao liquida
    # T0 = antoine_calc.get_vapor_temperature(P=P)[1] # Temperatura de vapor do agua, quando x1 = 0!
    # print(T0)
    # x_plot = x_range
    # y_plot = []
    # T_plot = []
    # for x in x_range:
    #     x_ = np.array([x, 1 - x])
    #     args = (x_, P, antoine_calc, pr_calc, mixture, nrtl_calc, )
    #     T = minimize(fun=bolha_T_FO, 
    #              x0=T0,
    #              args=args,
    #              method='NELDER-MEAD'
    #              )
    #     T0 = T.x[0]
    #     print(T0)
    #     y = bolha_T(T=T0, x=x_, P_sis=P, antoine_calc=antoine_calc, pr_calc=pr_calc, mixture=mixture, nrtl_calc=nrtl_calc)
    #     T_plot.append(T0)
    #     y_plot.append(y[0])