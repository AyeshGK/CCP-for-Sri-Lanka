import mesa
import random
import xlsxwriter
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet_wealth = workbook.add_worksheet()
worksheet_reward = workbook.add_worksheet()
class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, market_price, agent_type):
        super().__init__(unique_id, model)
        self.wealth = 35
        self.cash = 25
        self.shares = 1
        self.state_price = [0, 10]
        self.shares_avg = 10
        self.type = agent_type
        self.market_price = market_price
        self.reward = 0
        self.CCP_credit_max = 35*0.3
        self.CCP_credit= 0
        
        # credit_default list contains,
        # [going for a credit default or not , hold | buy or sell , if yes with whom , time left to the transaction ,share price]
        self.credit_default = ['No' , 0 , 0 , 0 ,10]

    def Zero_intel(self):
        prev_market_price = self.market_price
        # three states defined for the hold,buy,sell as [0,1,2]
        State = random.choice([0, 1, 2])
        if State == 0:

            return State, 0
        elif State == 1:
            price = random.randint(
                int(prev_market_price*0.8), int(prev_market_price*1.2))
            # print("buyz " + "and " + str(price))
            return State, price

        elif State == 2:
            price = random.randint(
                int(prev_market_price*0.8), int(prev_market_price*1.2))
            # print("sellz " + "and " + str(price))
            return State, price

    def zero_intel_reward(self):
        states = [0, 1, 2]
        reward = self.reward
        prev_market_price = self.market_price
        
        # three states defined for the hold,buy,sell as [0,1,2]
        random_State = random.choice(states)
        states.remove(self.state_price[0])
        random_state_1 = random.choice(states)
        price = random.randint(int(prev_market_price*0.8),
                               int(prev_market_price*1.2))
        if reward == 0:
            return random_State, price
        elif reward == 1:
            return self.state_price[0], price
        elif reward == -1:
            return random_state_1, price
        
    def update_credit_default(self):
        status = random.choice(['Yes','No'])
        state = random.choice([0,1,2])
        previous_credit_default =  self.credit_default
        updated_list = [status] + [state] + previous_credit_default[2:]
        self.credit_default = updated_list
    
    def Update_CCP_wealth(self):
            self.CCP_credit_max = self.wealth * (0.3)    
    

    def state_control(self):
        if self.type == "zero_intel":
            self.state_price = self.Zero_intel()
        elif self.type == "zero_intel_reward":
            self.state_price = self.zero_intel_reward()
            
        # change the state of credit_default
        self.update_credit_default()

    def step(self):
        # The agent's step will go here.
        self.state_control()
        self.Update_CCP_wealth()


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, Zero_agents, Reward_agents, market_price):
        self.Zero_agents = Zero_agents
        self.Reward_agents = Reward_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.market_price = market_price
        self.CCP_wealth = 1000
        
        # Create agents
        for i in range(self.Zero_agents):
            a = MoneyAgent(i, self, self.market_price, "zero_intel")
            self.schedule.add(a)
        for j in range(self.Reward_agents):
            a = MoneyAgent(j + self.Zero_agents, self,
                           self.market_price, "zero_intel_reward")
            self.schedule.add(a)

        self.datacollector = mesa.datacollection.DataCollector(
            agent_reporters={"Wealth": "state_price"})

        self.datacollector = DataCollector(
            model_reporters={
                'list_market_price': "market_price"})
    

    def market(self):
        market_price = self.market_price
        Agents = self.schedule.agents
        ccp_wealth = self.CCP_wealth
        
        def CCP(ccp_wealth,Agents_ID):

            if credit_default == 'Yes':
                ccp_wealth -= Agent.shares * Agent.market_price
            return ccp_wealth, Agents
        
        def credit_default(Agent):
            state = Agent.state
            time_left = Agent.time
            
            if time_left <= 0:
                CCP()
        
                


        def get_transaction_agents(Agents):

            def get_state_price(Agents):
                state_list = []
                for i in Agents:
                    state_list.append(i.state_price)
                return state_list

            def get_prices(Agents):
                state_prices = []
                for i in Agents:
                    state_prices.append(i.state_price[1])
                return state_prices

            prices = np.array(get_prices(Agents))
            states_price = np.array(get_state_price(Agents))

            transaction_agents = []

            for i in range(len(prices)):
                states_price_list = states_price.tolist()
                check = ([1, prices[i]] in states_price_list) and (
                    [2, prices[i]] in states_price_list)
                if(check):
                    buy_sell = [states_price_list.index(
                        [1, prices[i]]), states_price_list.index([2, prices[i]])]
                    transaction_agents.append(buy_sell)

            return transaction_agents

        def trading_transaction(agent_sell, agent_buy):

            # update selling agent
            agent_sell.shares = agent_sell.shares - 1
            agent_sell.state = 0
            agent_sell.cash = agent_sell.cash + (agent_sell.state_price[1])*1
            agent_sell.wealth = agent_sell.cash + \
                (agent_sell.state_price[1])*agent_sell.shares

            # update Buying agent
            agent_buy.shares = agent_buy.shares + 1
            agent_buy.state = 0
            agent_buy.cash = agent_buy.cash - (agent_buy.state_price[1])*1
            agent_buy.wealth = agent_buy.cash + \
                (agent_buy.state_price[1])*agent_buy.shares
            return
        
        def credit_transaction(agent_sell, agent_buy):
            # update selling agent
            agent_sell.shares = agent_sell.shares - 1
            agent_sell.state = 0
            agent_sell.cash = agent_sell.cash + (agent_sell.state_price[1])*1
            agent_sell.wealth = agent_sell.cash + \
                (agent_sell.state_price[1])*agent_sell.shares

            # update Buying agent
            agent_buy.shares = agent_buy.shares + 1
            agent_buy.state = 0
            agent_buy.CCP_credit = -(agent_buy.state_price[1])*1
            agent_buy.wealth = agent_buy.cash + \
                (agent_buy.state_price[1])*agent_buy.shares
        
        
            
        buy_Sell_agents = get_transaction_agents(Agents)
        for buy_Sell_agent in buy_Sell_agents:
            Agent_buy = Agents[buy_Sell_agent[0]]
            Agent_sell = Agents[buy_Sell_agent[1]]
            if (Agent_buy.cash > Agent_buy.state_price[1]) & (Agent_sell.shares > 0):
                trading_transaction(Agent_sell, Agent_buy)
                print_str = str(Agent_sell.unique_id)+" sold to " + str(
                    Agent_buy.unique_id) + " at price "+str((Agent_sell.state_price[1]))
                print(print_str)
                market_price = Agent_sell.state_price[1]
                self.market_price = market_price
            elif(Agent_buy.cash < Agent_buy.state_price[1]) & (Agent_sell.shares > 0) & (Agent_buy.CCP_credit_max > Agent_buy.cash):
                credit_transaction(Agent_sell, Agent_buy)
                market_price = Agent_sell.state_price[1]
                self.market_price = market_price
                self.CCP_wealth = self.CCP_wealth - market_price
                print("------------------------------------------")
                

        

    def Update_wealth(self):
        market_price = self.market_price
        Agents = self.schedule.agents

        for Agent in Agents:
            if Agent.type == "zero_intel_reward":
                if Agent.wealth > (Agent.cash + market_price*Agent.shares):
                    Agent.reward = 1
                elif Agent.wealth < (Agent.cash + market_price*Agent.shares):
                    Agent.reward = -1
                elif Agent.wealth == (Agent.cash + market_price*Agent.shares):
                    Agent.reward = 0
            Agent.wealth = Agent.cash + market_price

    def step(self):
        """Advance the model by one step."""
        self.market()
        self.Update_wealth()
        self.datacollector.collect(self)
        self.schedule.step()


Zero_agents = 15
Reward_agents = 15
iterations = 50
market_price = 10
model = MoneyModel(Zero_agents, Reward_agents, market_price)

list_avg_zero = []
list_avg_reward = []

plt.figure(figsize=(18, 6))

for i in range(iterations):
    print("Day" + str(i+1) + ":")
    model.step()

    agent_wealth = [a.wealth for a in model.schedule.agents]
    agent_reward = [a.reward for a in model.schedule.agents]
    agent_credit_default = [a.credit_default for a in model.schedule.agents]
    average_wealth_zero = sum(agent_wealth[:Zero_agents])/Zero_agents
    list_avg_zero.append(average_wealth_zero)
    average_wealth_reward = sum(agent_wealth[Zero_agents:Zero_agents+Reward_agents])/Reward_agents
    list_avg_reward.append(average_wealth_reward)
    print("average wealth Zero agents : ", sum(
        agent_wealth[:Zero_agents])/Zero_agents)
    print("average wealth Reward agents : ", sum(
        agent_wealth[Zero_agents:Zero_agents+Reward_agents])/Reward_agents)

    data_market = model.datacollector.get_model_vars_dataframe()
    data_market = data_market.rename(columns={data_market.columns[0]: 'data'})
    
    plt.subplot(131)
    plt.hist(agent_wealth, density=False,bins=10)
    plt.title("Histogram of Agent Wealth")
    plt.subplot(132)
    plt.plot(list_avg_zero,color='r',label='Zero Intel')
    plt.plot(list_avg_reward,color='g',label='Level 1 Intel')
    plt.title("Average wealth values of zero and reward")
    plt.suptitle("Agent Handling")
    plt.subplot(133)
    plt.plot(data_market["data"])
    plt.title("Market price")
    plt.suptitle('Stock market')
    plt.pause(0.5)
    plt.clf()
    worksheet_wealth.write_row(i, 0, agent_wealth)
    worksheet_reward.write_row(i, 0, agent_reward)
    print(agent_credit_default)



plt.show()
workbook.close()