#include <atomic>
#include <stdexcept>

class TicketOffice
{
public:
    explicit TicketOffice(int numTickets)
            : m_numTickets(numTickets)
    {
    }

    TicketOffice(const TicketOffice &) = delete;

    TicketOffice &operator=(const TicketOffice &) = delete;

    int SellTickets(int ticketsToBuy)
    {
        if (ticketsToBuy <= 0)
        {
            throw std::invalid_argument("ticketsToBuy must be positive");
        }

        int currentTickets = m_numTickets.load(std::memory_order_relaxed);

        while (true)
        {
            if (currentTickets < ticketsToBuy)
            {
                if (currentTickets == 0)
                {
                    return 0;
                }
                ticketsToBuy = currentTickets;
            }

            if (m_numTickets.compare_exchange_weak(
                    currentTickets,
                    currentTickets - ticketsToBuy,
                    std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                return ticketsToBuy;
            }
        }
    }

    [[nodiscard]] int GetTicketsLeft() const noexcept
    {
        return m_numTickets.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int> m_numTickets;
};
